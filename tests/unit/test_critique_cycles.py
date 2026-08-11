"""Persisted, finite critic-cycle contracts at the database and decision boundaries."""

from __future__ import annotations

import json
import multiprocessing
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

import pytest

import invoice_agents.db.core as core_module
from invoice_agents.agents import decision_rules
from invoice_agents.config import Settings
from invoice_agents.db.core import (
    DatabaseKind,
    _migration_resources,
    connect_database,
    verify_database,
)
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import DatabaseVerificationError, InvoiceAgentsError
from invoice_agents.models import (
    CaseStatus,
    Critique,
    DecisionKind,
    FinalDecision,
    ReviewRequest,
    SourceArtifact,
)
from invoice_agents.observability.audit import AuditRecorder


def _source(tmp_path: Path, case_id: str) -> SourceArtifact:
    return SourceArtifact(
        source_id=f"src_{case_id}",
        canonical_path=(tmp_path / f"{case_id}.txt").resolve(),
        sha256="a" * 64,
        source_format="txt",
        size_bytes=1,
        modified_at=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
    )


def _claimed_case(
    settings: Settings,
    tmp_path: Path,
    case_id: str,
) -> tuple[WorkflowStore, ExecutionClaim]:
    source = _source(tmp_path, case_id)
    store = WorkflowStore(settings)
    store.register_source(source)
    store.create_case(case_id, source, datetime(2026, 8, 10, 12, 1, tzinfo=UTC))
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    return store, claim


def _critique(
    *,
    cycle: int = 1,
    responds_to_critique_id: str | None = None,
    challenged_findings: list[str] | None = None,
    requested_follow_up: list[str] | None = None,
    supported_findings: list[str] | None = None,
    missing_evidence: list[str] | None = None,
    disposition: DecisionKind = DecisionKind.HOLD,
    rationale: list[str] | None = None,
    follow_up_responses: list[dict[str, object]] | None = None,
) -> Critique:
    payload: dict[str, object] = {
        "cycle": cycle,
        "responds_to_critique_id": responds_to_critique_id,
        "supported_findings": (
            supported_findings
            if supported_findings is not None
            else ["persisted evidence reviewed"]
        ),
        "challenged_findings": challenged_findings or [],
        "missing_evidence": missing_evidence or [],
        "requested_follow_up": requested_follow_up or [],
        "recommended_disposition": disposition,
        "rationale": rationale or ["finite persisted critique cycle"],
    }
    if follow_up_responses is not None:
        payload["follow_up_responses"] = follow_up_responses
    return Critique.model_validate(payload)


def _record_follow_up_evidence(
    settings: Settings,
    case_id: str,
    requested_item: str,
) -> str:
    return AuditRecorder(settings.workflow_db, case_id).record(
        "tool.critic_line_recompute",
        {
            "requested_item": requested_item,
            "quantity": "2",
            "unit_price": "5.00",
            "line_total": "10.00",
        },
        agent_name="independent_critic_agent",
    )


def _follow_up_response(
    requested_item: str,
    outcome: str,
    evidence_event_id: str,
) -> dict[str, object]:
    return {
        "requested_item": requested_item,
        "outcome": outcome,
        "evidence_event_ids": [evidence_event_id],
    }


def _assert_cycle_complete(case_id: str, store: WorkflowStore) -> None:
    checker = vars(decision_rules).get("assert_critique_cycle_complete")
    assert callable(checker), "decision rules do not enforce persisted critique-cycle completion"
    checker(case_id, store)


def test_workflow_v7_installs_exact_critique_cycle_schema(settings: Settings) -> None:
    resources = tuple(
        resource.name
        for resource in _migration_resources(DatabaseKind.WORKFLOW)
        if resource.name[:3].isdigit()
    )
    with connect_database(settings.workflow_db, read_only=True) as connection:
        versions = tuple(
            int(row[0])
            for row in connection.execute("SELECT version FROM schema_version ORDER BY version")
        )
        columns = tuple(
            (
                str(row[1]),
                str(row[2]),
                int(row[3]),
                None if row[4] is None else str(row[4]),
                int(row[5]),
            )
            for row in connection.execute("PRAGMA table_info(critique_results)")
        )
        foreign_keys = {
            (
                str(row[3]),
                str(row[2]),
                str(row[4]),
            )
            for row in connection.execute("PRAGMA foreign_key_list(critique_results)")
        }
        indexes = {
            str(row[1]): (int(row[2]), str(row[3]), int(row[4]))
            for row in connection.execute("PRAGMA index_list(critique_results)")
        }
        index_columns = tuple(
            str(row[2])
            for row in connection.execute(
                "PRAGMA index_info(idx_critique_results_case_cycle)"
            )
        )

    assert resources[-1] == "007_critique_cycles.sql"
    assert versions == (1, 2, 3, 4, 5, 6, 7)
    assert columns == (
        ("critique_id", "TEXT", 0, None, 1),
        ("case_id", "TEXT", 1, None, 0),
        ("payload_json", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
        ("execution_generation", "INTEGER", 1, "0", 0),
        ("cycle", "INTEGER", 1, "1", 0),
        ("responds_to_critique_id", "TEXT", 0, None, 0),
    )
    assert (
        "responds_to_critique_id",
        "critique_results",
        "critique_id",
    ) in foreign_keys
    assert indexes["idx_critique_results_case_cycle"] == (1, "c", 0)
    assert index_columns == ("case_id", "cycle")


def test_strict_manifest_rejects_removed_critique_cycle_uniqueness(settings: Settings) -> None:
    with connect_database(settings.workflow_db) as connection:
        connection.execute("DROP INDEX idx_critique_results_case_cycle")
        connection.commit()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=settings,
        )

    assert excinfo.value.stop_reason == "DATABASE_SCHEMA_MISMATCH"


def _empty_v6_workflow(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> tuple[Path, Settings]:
    path = tmp_path / filename
    packaged = core_module._migration_resources(DatabaseKind.WORKFLOW)
    assert [resource.name[:3] for resource in packaged] == [
        "001",
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
    ]
    original_resources = core_module._migration_resources

    def resources_through_v6(kind: DatabaseKind) -> list[object]:
        resources = original_resources(kind)
        return resources[:-1] if kind is DatabaseKind.WORKFLOW else list(resources)

    legacy_settings = settings.model_copy(
        update={
            "workflow_db": path,
            "source_archive_dir": tmp_path / f"{path.stem}-sources",
        }
    )
    monkeypatch.setattr(core_module, "_migration_resources", resources_through_v6)
    assert core_module._migrate_database_in_process(
        path,
        DatabaseKind.WORKFLOW,
        settings=legacy_settings,
    ) == [1, 2, 3, 4, 5, 6]
    monkeypatch.setattr(core_module, "_migration_resources", original_resources)
    return path, legacy_settings


def test_v7_upgrade_archives_every_legacy_critique_and_promotes_exact_latest(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, legacy_settings = _empty_v6_workflow(
        settings,
        tmp_path,
        monkeypatch,
        "workflow-v6-duplicate-critiques.db",
    )

    old_at = datetime(2026, 8, 10, 12, 2, tzinfo=UTC).isoformat()
    latest_at = datetime(2026, 8, 10, 12, 3, tzinfo=UTC).isoformat()
    legacy_payloads = (
        json.dumps(
            {
                "supported_findings": ["first legacy run"],
                "challenged_findings": [],
                "missing_evidence": [],
                "requested_follow_up": [],
                "recommended_disposition": "HOLD",
                "rationale": ["superseded by a normal resume"],
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "supported_findings": ["latest legacy resumed run"],
                "challenged_findings": [],
                "missing_evidence": [],
                "requested_follow_up": [],
                "recommended_disposition": "APPROVE",
                "rationale": ["this was the pre-v7 authoritative latest row"],
            },
            sort_keys=True,
        ),
    )
    with connect_database(path) as connection:
        connection.execute(
            "INSERT INTO cases(case_id, status, started_at, updated_at) "
            "VALUES ('case_legacy_duplicate_critiques', 'INCOMPLETE', ?, ?)",
            (old_at, old_at),
        )
        connection.executemany(
            "INSERT INTO critique_results("
            "critique_id, case_id, payload_json, created_at, execution_generation"
            ") VALUES (?, 'case_legacy_duplicate_critiques', ?, ?, ?)",
            (
                ("crit_legacy_first", legacy_payloads[0], old_at, 1),
                ("crit_legacy_latest", legacy_payloads[1], latest_at, 2),
            ),
        )
        connection.commit()

    assert core_module._migrate_database_in_process(
        path,
        DatabaseKind.WORKFLOW,
        settings=legacy_settings,
    ) == [7]
    verify_database(
        path,
        DatabaseKind.WORKFLOW,
        settings=legacy_settings,
    )
    with connect_database(path, read_only=True) as connection:
        active_row = connection.execute(
            "SELECT critique_id, payload_json, execution_generation, cycle, "
            "responds_to_critique_id FROM critique_results"
        ).fetchone()
        archived = [
            tuple(row)
            for row in connection.execute(
                "SELECT source_rowid, critique_id, case_id, payload_json, created_at, "
                "execution_generation, migration_role "
                "FROM legacy_critique_history ORDER BY source_rowid"
            )
        ]

    assert active_row is not None
    active = tuple(active_row)
    assert active == (
        "crit_legacy_latest",
        legacy_payloads[1],
        2,
        1,
        None,
    )
    assert archived == [
        (
            1,
            "crit_legacy_first",
            "case_legacy_duplicate_critiques",
            legacy_payloads[0],
            old_at,
            1,
            "HISTORICAL_SUPERSEDED",
        ),
        (
            2,
            "crit_legacy_latest",
            "case_legacy_duplicate_critiques",
            legacy_payloads[1],
            latest_at,
            2,
            "PROMOTED_CYCLE_ONE",
        ),
    ]
    immutable_statements = (
        "UPDATE legacy_critique_history SET payload_json = '{}'",
        "DELETE FROM legacy_critique_history",
        "INSERT INTO legacy_critique_history("
        "source_rowid, critique_id, case_id, payload_json, created_at, "
        "execution_generation, migration_role) "
        "SELECT source_rowid + 100, critique_id || '_copy', case_id, payload_json, "
        "created_at, execution_generation, 'HISTORICAL_SUPERSEDED' "
        "FROM legacy_critique_history LIMIT 1",
    )
    for statement in immutable_statements:
        with connect_database(path) as connection:
            with pytest.raises(
                sqlite3.IntegrityError,
                match="LEGACY_CRITIQUE_HISTORY_IMMUTABLE",
            ):
                connection.execute(statement)
            connection.rollback()


def test_v7_upgrade_rolls_back_an_equal_latest_legacy_critique_ambiguity(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, legacy_settings = _empty_v6_workflow(
        settings,
        tmp_path,
        monkeypatch,
        "workflow-v6-ambiguous-critiques.db",
    )
    tied_at = datetime(2026, 8, 10, 12, 2, tzinfo=UTC).isoformat()
    payload = json.dumps(
        {
            "supported_findings": ["legacy evidence"],
            "challenged_findings": [],
            "missing_evidence": [],
            "requested_follow_up": [],
            "recommended_disposition": "HOLD",
            "rationale": ["the legacy created_at tie is not authoritative"],
        },
        sort_keys=True,
    )
    with connect_database(path) as connection:
        connection.execute(
            "INSERT INTO cases(case_id, status, started_at, updated_at) "
            "VALUES ('case_legacy_critique_tie', 'INCOMPLETE', ?, ?)",
            (tied_at, tied_at),
        )
        connection.executemany(
            "INSERT INTO critique_results("
            "critique_id, case_id, payload_json, created_at, execution_generation"
            ") VALUES (?, 'case_legacy_critique_tie', ?, ?, ?)",
            (
                ("crit_legacy_tied_a", payload, tied_at, 1),
                ("crit_legacy_tied_b", payload, tied_at, 2),
            ),
        )
        connection.commit()
    before = path.read_bytes()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        core_module._migrate_database_in_process(
            path,
            DatabaseKind.WORKFLOW,
            settings=legacy_settings,
        )

    assert excinfo.value.stop_reason == "LEGACY_CRITIQUE_HISTORY_AMBIGUOUS"
    assert path.read_bytes() == before
    with connect_database(path, read_only=True) as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0] == 6
        assert connection.execute(
            "SELECT COUNT(*) FROM critique_results WHERE case_id = 'case_legacy_critique_tie'"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'legacy_critique_history'"
        ).fetchone() is None


def test_legacy_payload_and_default_relational_columns_remain_cycle_one(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_legacy_critique")
    legacy_payload = {
        "supported_findings": ["legacy evidence reviewed"],
        "challenged_findings": [],
        "missing_evidence": [],
        "requested_follow_up": [],
        "recommended_disposition": "HOLD",
        "rationale": ["legacy payload has no cycle keys"],
    }
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "INSERT INTO critique_results("
            "critique_id, case_id, payload_json, created_at, execution_generation"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                "crit_legacy",
                claim.case_id,
                json.dumps(legacy_payload),
                datetime(2026, 8, 10, 12, 2, tzinfo=UTC).isoformat(),
                claim.generation,
            ),
        )
        row = connection.execute(
            "SELECT cycle, responds_to_critique_id FROM critique_results "
            "WHERE critique_id = 'crit_legacy'"
        ).fetchone()
        connection.commit()

    critiques = store.list_critiques(claim.case_id)
    assert row is not None
    assert tuple(row) == (1, None)
    assert [(item.cycle, item.responds_to_critique_id) for item in critiques] == [(1, None)]


def test_save_and_list_critiques_preserve_cycle_order_and_exact_parent(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_ordered_critiques")
    first_id = store.save_critique(
        claim.case_id,
        _critique(challenged_findings=["inventory mapping needs recheck"]),
        claim,
    )
    store.release_case_execution(claim)
    resumed_claim = store.claim_case_execution(
        claim.case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    assert resumed_claim.generation == claim.generation + 1
    evidence_event_id = _record_follow_up_evidence(
        settings,
        resumed_claim.case_id,
        "inventory mapping needs recheck",
    )
    second_id = store.save_critique(
        resumed_claim.case_id,
        _critique(
            cycle=2,
            responds_to_critique_id=first_id,
            challenged_findings=["inventory mapping needs recheck"],
            follow_up_responses=[
                _follow_up_response(
                    "inventory mapping needs recheck",
                    "CHALLENGED",
                    evidence_event_id,
                )
            ],
        ),
        resumed_claim,
    )

    critiques = store.list_critiques(resumed_claim.case_id)
    with connect_database(settings.workflow_db, read_only=True) as connection:
        ids = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT critique_id FROM critique_results WHERE case_id = ? ORDER BY cycle",
                (resumed_claim.case_id,),
            )
        )

    assert ids == (first_id, second_id)
    assert [item.cycle for item in critiques] == [1, 2]
    assert critiques[0].responds_to_critique_id is None
    assert critiques[1].responds_to_critique_id == first_id
    assert store.load_current_critique(resumed_claim).cycle == 2


def test_cycle_one_cannot_claim_a_parent_at_the_store_boundary(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_cycle_one_parent")

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_critique(
            claim.case_id,
            _critique(cycle=1, responds_to_critique_id="crit_impossible"),
            claim,
        )

    assert excinfo.value.stop_reason == "CRITIQUE_RESPONSE_INVALID"
    assert store.list_critiques(claim.case_id) == []


@pytest.mark.parametrize("parent_kind", ["missing", "different-case"])
def test_cycle_two_requires_the_exact_same_case_cycle_one_identity(
    settings: Settings,
    tmp_path: Path,
    parent_kind: str,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, f"case_response_{parent_kind}")
    first_id = store.save_critique(
        claim.case_id,
        _critique(challenged_findings=["line total needs recheck"]),
        claim,
    )
    parent_id = "crit_missing"
    if parent_kind == "different-case":
        other_store, other_claim = _claimed_case(settings, tmp_path, "case_response_other")
        parent_id = other_store.save_critique(other_claim.case_id, _critique(), other_claim)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_critique(
            claim.case_id,
            _critique(
                cycle=2,
                responds_to_critique_id=parent_id,
                challenged_findings=["line total needs recheck"],
            ),
            claim,
        )

    assert excinfo.value.stop_reason == "CRITIQUE_RESPONSE_INVALID"
    assert [item.cycle for item in store.list_critiques(claim.case_id)] == [1]
    assert first_id.startswith("crit_")


def test_cycle_two_covers_requested_follow_up_across_the_exact_structured_union(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_follow_up_union")
    requested = [
        "recheck exact inventory row",
        "recompute line total",
        "locate missing purchase order",
    ]
    first_id = store.save_critique(
        claim.case_id,
        _critique(requested_follow_up=requested),
        claim,
    )
    evidence_event_ids = [
        _record_follow_up_evidence(settings, claim.case_id, requested_item)
        for requested_item in requested
    ]

    second_id = store.save_critique(
        claim.case_id,
        _critique(
            cycle=2,
            responds_to_critique_id=first_id,
            supported_findings=[requested[0]],
            challenged_findings=[requested[1]],
            missing_evidence=[requested[2]],
            follow_up_responses=[
                _follow_up_response(requested[0], "SUPPORTED", evidence_event_ids[0]),
                _follow_up_response(requested[1], "CHALLENGED", evidence_event_ids[1]),
                _follow_up_response(requested[2], "MISSING", evidence_event_ids[2]),
            ],
        ),
        claim,
    )

    _assert_cycle_complete(claim.case_id, store)
    with connect_database(settings.workflow_db, read_only=True) as connection:
        persisted_responses = [
            tuple(row)
            for row in connection.execute(
                "SELECT requested_item, outcome, evidence_event_id "
                "FROM critique_follow_up_evidence WHERE critique_id = ? "
                "ORDER BY requested_item",
                (second_id,),
            )
        ]
    assert persisted_responses == sorted(
        [
            (requested[0], "SUPPORTED", evidence_event_ids[0]),
            (requested[1], "CHALLENGED", evidence_event_ids[1]),
            (requested[2], "MISSING", evidence_event_ids[2]),
        ]
    )


def test_cycle_two_cannot_satisfy_follow_up_by_echoing_items_without_evidence(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_follow_up_echo")
    requested_item = "recompute exact line extension"
    first_id = store.save_critique(
        claim.case_id,
        _critique(requested_follow_up=[requested_item]),
        claim,
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_critique(
            claim.case_id,
            _critique(
                cycle=2,
                responds_to_critique_id=first_id,
                supported_findings=[requested_item],
            ),
            claim,
        )

    assert excinfo.value.stop_reason == "CRITIQUE_FOLLOW_UP_EVIDENCE_REQUIRED"
    assert [item.cycle for item in store.list_critiques(claim.case_id)] == [1]


@pytest.mark.parametrize(
    "evidence_boundary",
    ["different-case", "before-parent", "wrong-type"],
)
def test_cycle_two_rejects_an_unbound_or_preexisting_audit_event(
    settings: Settings,
    tmp_path: Path,
    evidence_boundary: str,
) -> None:
    store, claim = _claimed_case(
        settings,
        tmp_path,
        f"case_follow_up_event_{evidence_boundary}",
    )
    requested_item = "recompute exact line extension"
    if evidence_boundary == "before-parent":
        evidence_event_id = _record_follow_up_evidence(
            settings,
            claim.case_id,
            requested_item,
        )
    first_id = store.save_critique(
        claim.case_id,
        _critique(requested_follow_up=[requested_item]),
        claim,
    )
    if evidence_boundary == "different-case":
        other_store, other_claim = _claimed_case(
            settings,
            tmp_path,
            "case_follow_up_event_owner",
        )
        evidence_event_id = _record_follow_up_evidence(
            settings,
            other_claim.case_id,
            requested_item,
        )
        assert other_store.list_critiques(other_claim.case_id) == []
    elif evidence_boundary == "wrong-type":
        assert evidence_boundary == "wrong-type"
        evidence_event_id = AuditRecorder(settings.workflow_db, claim.case_id).record(
            "tool.case_metadata",
            {"requested_item": requested_item},
            agent_name="case_coordinator",
        )
    else:
        assert evidence_boundary == "before-parent"

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_critique(
            claim.case_id,
            _critique(
                cycle=2,
                responds_to_critique_id=first_id,
                supported_findings=[requested_item],
                follow_up_responses=[
                    _follow_up_response(
                        requested_item,
                        "SUPPORTED",
                        evidence_event_id,
                    )
                ],
            ),
            claim,
        )

    assert excinfo.value.stop_reason == "CRITIQUE_FOLLOW_UP_EVIDENCE_INVALID"
    assert [item.cycle for item in store.list_critiques(claim.case_id)] == [1]


@pytest.mark.parametrize(
    "mutation_sql",
    [
        "UPDATE events SET payload_json = '{}' WHERE event_id = ?",
        "DELETE FROM events WHERE event_id = ?",
    ],
    ids=["update", "delete"],
)
def test_bound_critique_follow_up_events_are_immutable(
    settings: Settings,
    tmp_path: Path,
    mutation_sql: str,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_follow_up_event_immutable")
    requested_item = "recompute exact line extension"
    first_id = store.save_critique(
        claim.case_id,
        _critique(requested_follow_up=[requested_item]),
        claim,
    )
    evidence_event_id = _record_follow_up_evidence(
        settings,
        claim.case_id,
        requested_item,
    )
    store.save_critique(
        claim.case_id,
        _critique(
            cycle=2,
            responds_to_critique_id=first_id,
            supported_findings=[requested_item],
            follow_up_responses=[
                _follow_up_response(
                    requested_item,
                    "SUPPORTED",
                    evidence_event_id,
                )
            ],
        ),
        claim,
    )

    with connect_database(settings.workflow_db) as connection:
        with pytest.raises(
            sqlite3.IntegrityError,
            match="CRITIQUE_FOLLOW_UP_EVIDENCE_IMMUTABLE",
        ):
            connection.execute(mutation_sql, (evidence_event_id,))
        connection.rollback()
    with connect_database(settings.workflow_db, read_only=True) as connection:
        assert connection.execute(
            "SELECT 1 FROM events WHERE event_id = ?",
            (evidence_event_id,),
        ).fetchone() is not None


@pytest.mark.parametrize(
    ("addressed", "rationale"),
    [
        (["Recompute line total"], ["finite persisted critique cycle"]),
        ([], ["recompute line total"]),
    ],
    ids=["case-normalized-text", "rationale-only"],
)
def test_cycle_two_coverage_is_exact_and_only_the_three_structured_lists_count(
    settings: Settings,
    tmp_path: Path,
    addressed: list[str],
    rationale: list[str],
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_follow_up_exact")
    requested = ["recompute line total"]
    first_id = store.save_critique(
        claim.case_id,
        _critique(requested_follow_up=requested),
        claim,
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_critique(
            claim.case_id,
            _critique(
                cycle=2,
                responds_to_critique_id=first_id,
                supported_findings=addressed,
                rationale=rationale,
            ),
            claim,
        )

    assert excinfo.value.stop_reason == "CRITIQUE_FOLLOW_UP_UNADDRESSED"
    assert [item.cycle for item in store.list_critiques(claim.case_id)] == [1]


def test_cycle_two_omission_of_one_requested_follow_up_fails_closed(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_missing_follow_up")
    requested = ["recheck exact inventory row", "recompute line total"]
    first_id = store.save_critique(
        claim.case_id,
        _critique(requested_follow_up=requested),
        claim,
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_critique(
            claim.case_id,
            _critique(
                cycle=2,
                responds_to_critique_id=first_id,
                supported_findings=[requested[0]],
            ),
            claim,
        )

    assert excinfo.value.stop_reason == "CRITIQUE_FOLLOW_UP_UNADDRESSED"
    assert [item.cycle for item in store.list_critiques(claim.case_id)] == [1]


@pytest.mark.parametrize(
    ("cycle_one_update", "required_item"),
    [
        (
            {"challenged_findings": ["inventory mapping is disputed"]},
            "inventory mapping is disputed",
        ),
        (
            {"missing_evidence": ["purchase order is absent"]},
            "purchase order is absent",
        ),
    ],
    ids=["challenged-finding", "missing-evidence"],
)
def test_cycle_two_must_explicitly_account_for_every_fact_that_forced_follow_up(
    settings: Settings,
    tmp_path: Path,
    cycle_one_update: dict[str, list[str]],
    required_item: str,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_required_follow_up_fact")
    first_id = store.save_critique(claim.case_id, _critique(**cycle_one_update), claim)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_critique(
            claim.case_id,
            _critique(
                cycle=2,
                responds_to_critique_id=first_id,
                supported_findings=["an unrelated check completed"],
            ),
            claim,
        )

    assert required_item not in store.list_critiques(claim.case_id)[0].supported_findings
    assert excinfo.value.stop_reason == "CRITIQUE_FOLLOW_UP_UNADDRESSED"
    assert [item.cycle for item in store.list_critiques(claim.case_id)] == [1]


@pytest.mark.parametrize(
    "cycle_one_update",
    [
        {"challenged_findings": ["inventory mapping is disputed"]},
        {"requested_follow_up": ["recheck exact inventory row"]},
    ],
    ids=["challenged-finding", "requested-follow-up"],
)
def test_persisted_cycle_one_disagreement_blocks_finalization(
    settings: Settings,
    tmp_path: Path,
    cycle_one_update: dict[str, list[str]],
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_cycle_one_blocked")
    store.save_critique(claim.case_id, _critique(**cycle_one_update), claim)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        _assert_cycle_complete(claim.case_id, store)

    assert excinfo.value.stop_reason == "CRITIQUE_FOLLOW_UP_REQUIRED"


def test_incomplete_critique_cycle_cannot_cross_the_human_review_boundary(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_review_cycle_gate")
    critique = _critique(requested_follow_up=["recompute amount"])
    store.save_critique(claim.case_id, critique, claim)
    review = ReviewRequest(
        review_id="rev_incomplete_critique",
        case_id=claim.case_id,
        status="PENDING",
        reasons=["critic requested follow-up"],
        amount=None,
        source=_source(tmp_path, claim.case_id),
        evidence_bundle={},
        agent_recommendation=DecisionKind.HOLD,
        agent_rationale=["hold until follow-up completes"],
        critic=critique,
        critic_disagreement_reason=None,
        questions=["Was the requested follow-up completed?"],
        created_at=datetime(2026, 8, 10, 12, 2, tzinfo=UTC),
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_review(review, claim)

    assert excinfo.value.stop_reason == "CRITIQUE_FOLLOW_UP_REQUIRED"
    assert store.list_reviews(pending_only=False) == []


def test_clean_cycle_one_is_complete_without_fabricating_a_second_cycle(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_clean_cycle_one")
    store.save_critique(claim.case_id, _critique(), claim)

    _assert_cycle_complete(claim.case_id, store)

    assert [item.cycle for item in store.list_critiques(claim.case_id)] == [1]


def test_clean_cycle_one_rejects_an_unnecessary_prompt_requested_cycle_two(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_unnecessary_cycle_two")
    first_id = store.save_critique(claim.case_id, _critique(), claim)

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_critique(
            claim.case_id,
            _critique(
                cycle=2,
                responds_to_critique_id=first_id,
            ),
            claim,
        )

    assert excinfo.value.stop_reason == "CRITIQUE_FOLLOW_UP_NOT_REQUIRED"
    assert [item.cycle for item in store.list_critiques(claim.case_id)] == [1]


def test_cycle_two_cannot_request_an_unpersistable_third_cycle(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_cycle_two_follow_up")
    first_id = store.save_critique(
        claim.case_id,
        _critique(requested_follow_up=["recheck exact inventory row"]),
        claim,
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_critique(
            claim.case_id,
            _critique(
                cycle=2,
                responds_to_critique_id=first_id,
                supported_findings=["recheck exact inventory row"],
                requested_follow_up=["perform a third recheck"],
            ),
            claim,
        )

    assert excinfo.value.stop_reason == "CRITIQUE_CYCLE_LIMIT"
    assert [item.cycle for item in store.list_critiques(claim.case_id)] == [1]


def test_third_critique_fails_with_exact_limit_and_preserves_two_rows(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_third_critique")
    first_id = store.save_critique(
        claim.case_id,
        _critique(challenged_findings=["recompute amount"]),
        claim,
    )
    evidence_event_id = _record_follow_up_evidence(
        settings,
        claim.case_id,
        "recompute amount",
    )
    second_id = store.save_critique(
        claim.case_id,
        _critique(
            cycle=2,
            responds_to_critique_id=first_id,
            challenged_findings=["recompute amount"],
            follow_up_responses=[
                _follow_up_response(
                    "recompute amount",
                    "CHALLENGED",
                    evidence_event_id,
                )
            ],
        ),
        claim,
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_critique(
            claim.case_id,
            _critique(cycle=2, responds_to_critique_id=first_id),
            claim,
        )

    assert excinfo.value.stop_reason == "CRITIQUE_CYCLE_LIMIT"
    assert [item.cycle for item in store.list_critiques(claim.case_id)] == [1, 2]
    assert second_id.startswith("crit_")


def _save_cycle_two_worker(
    workflow_db: str,
    inventory_db: str,
    claim: ExecutionClaim,
    first_id: str,
    requested: list[str],
    evidence_event_id: str,
    start: Any,
    result: Any,
) -> None:
    """One independently spawned contender; every outcome is sent before exit."""

    try:
        store = WorkflowStore(
            Settings(
                xai_api_key="test-only-not-a-real-key",
                workflow_db=Path(workflow_db),
                inventory_db=Path(inventory_db),
            )
        )
        store.require_current_execution_claim(claim)
        result.send(("ready", None))
        if not start.wait(timeout=10):
            result.send(("harness", "start-timeout"))
            return
        try:
            record_id = store.save_critique(
                claim.case_id,
                _critique(
                    cycle=2,
                    responds_to_critique_id=first_id,
                    supported_findings=requested,
                    follow_up_responses=[
                        _follow_up_response(
                            requested[0],
                            "SUPPORTED",
                            evidence_event_id,
                        )
                    ],
                ),
                claim,
            )
        except InvoiceAgentsError as exc:
            result.send(("error", exc.stop_reason))
            return
        result.send(("saved", record_id))
    except BaseException as exc:
        result.send(("unexpected", f"{type(exc).__name__}: {exc}"))
    finally:
        result.close()


def test_concurrent_cycle_two_writers_commit_exactly_one_response(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_concurrent_critique")
    requested = ["recompute amount"]
    first_id = store.save_critique(
        claim.case_id,
        _critique(requested_follow_up=requested),
        claim,
    )
    evidence_event_id = _record_follow_up_evidence(
        settings,
        claim.case_id,
        requested[0],
    )
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    result_pairs = [context.Pipe(duplex=False) for _ in range(2)]
    processes = [
        context.Process(
            target=_save_cycle_two_worker,
            args=(
                str(settings.workflow_db),
                str(settings.inventory_db),
                claim,
                first_id,
                requested,
                evidence_event_id,
                start,
                child_result,
            ),
        )
        for _parent_result, child_result in result_pairs
    ]
    started: list[multiprocessing.Process] = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        for _parent_result, child_result in result_pairs:
            child_result.close()
        ready_deadline = monotonic() + 10
        ready_receipts = [
            (
                parent_result.recv()
                if parent_result.poll(max(0.0, ready_deadline - monotonic()))
                else ("harness", "ready-timeout")
            )
            for parent_result, _child_result in result_pairs
        ]
        assert ready_receipts == [("ready", None), ("ready", None)]
        start.set()
        deadline = monotonic() + 10
        for process in started:
            process.join(timeout=max(0.0, deadline - monotonic()))
        alive = [process for process in started if process.is_alive()]
        assert alive == [], "concurrent critique writers did not terminate within 10 seconds"
        assert [process.exitcode for process in started] == [0, 0]
        outcomes = [
            parent_result.recv() if parent_result.poll(1) else ("harness", "missing-result")
            for parent_result, _child_result in result_pairs
        ]
    finally:
        for process in started:
            if process.is_alive():
                process.terminate()
        for process in started:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        for parent_result, child_result in result_pairs:
            parent_result.close()
            child_result.close()

    assert sorted(status for status, _detail in outcomes) == ["error", "saved"]
    assert [detail for status, detail in outcomes if status == "error"] == [
        "CRITIQUE_CYCLE_LIMIT"
    ]
    assert [item.cycle for item in store.list_critiques(claim.case_id)] == [1, 2]


def test_final_decision_persistence_checks_stored_cycle_before_other_decision_rules(
    settings: Settings,
    tmp_path: Path,
) -> None:
    store, claim = _claimed_case(settings, tmp_path, "case_final_cycle_gate")
    store.save_critique(
        claim.case_id,
        _critique(requested_follow_up=["recompute amount"]),
        claim,
    )
    final = FinalDecision(
        decision=DecisionKind.HOLD,
        reasons=["follow-up is not complete"],
        critic_disposition=DecisionKind.HOLD,
        payment_eligible=False,
    )

    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.save_final_decision(claim.case_id, final, claim)

    assert excinfo.value.stop_reason == "CRITIQUE_FOLLOW_UP_REQUIRED"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        final_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM final_decisions WHERE case_id = ?",
                (claim.case_id,),
            ).fetchone()[0]
        )
    assert final_count == 0


@pytest.mark.parametrize(
    ("critique_id", "cycle", "responds_to_critique_id"),
    [
        ("crit_cycle_three", 3, None),
        ("crit_cycle_one_with_parent", 1, "crit_cycle_one_with_parent"),
        ("crit_cycle_two_without_parent", 2, None),
        ("crit_cycle_two_missing_parent", 2, "crit_nonexistent"),
    ],
    ids=[
        "cycle-three",
        "cycle-one-with-parent",
        "cycle-two-without-parent",
        "cycle-two-missing-parent",
    ],
)
def test_relational_checks_reject_invalid_cycle_shape_without_store_bypass(
    settings: Settings,
    tmp_path: Path,
    critique_id: str,
    cycle: int,
    responds_to_critique_id: str | None,
) -> None:
    _store, claim = _claimed_case(settings, tmp_path, "case_relational_cycle_check")
    payload = _critique().model_dump_json()

    with connect_database(settings.workflow_db) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO critique_results("
                "critique_id, case_id, payload_json, created_at, execution_generation, cycle, "
                "responds_to_critique_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    critique_id,
                    claim.case_id,
                    payload,
                    datetime(2026, 8, 10, 12, 2, tzinfo=UTC).isoformat(),
                    claim.generation,
                    cycle,
                    responds_to_critique_id,
                ),
            )
        connection.rollback()
