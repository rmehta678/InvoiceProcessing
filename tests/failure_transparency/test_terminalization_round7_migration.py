"""Round 7 migration regressions for strict historical recovery payloads."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import invoice_agents.db.core as db_core
from invoice_agents.db.core import DatabaseKind, connect_database, migrate_database
from invoice_agents.errors import DatabaseVerificationError, ErrorCategory
from invoice_agents.models import (
    CaseResult,
    CaseStatus,
    Critique,
    DecisionKind,
    ErrorRecord,
    FinalDecision,
    Money,
    PaymentResult,
    PaymentStatus,
    ReviewRequest,
    SourceArtifact,
    UsageSummary,
)

_STARTED_AT = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
_PREVIOUS_FINISHED_AT = datetime(2026, 8, 9, 12, 1, tzinfo=UTC)
_EXPIRED_LEASE = datetime(2026, 8, 9, 12, 4, tzinfo=UTC)
_RECOVERED_AT = datetime(2026, 8, 9, 12, 5, tzinfo=UTC)
_TOKEN_SUFFIXES = {
    "case_round7_idle_zero": "01" * 16,
    "case_round7_running": "ab" * 16,
    "case_round7_forged": "cd" * 16,
    "case_round7_large_usage": "34" * 16,
}


def _workflow_resources() -> list[Any]:
    resources = db_core._migration_resources(DatabaseKind.WORKFLOW)
    assert [resource.name for resource in resources] == [
        "001_initial.sql",
        "002_review_sequence.sql",
        "003_execution_fencing.sql",
        "004_execution_token_grammar.sql",
        "005_result_artifact_bindings.sql",
    ]
    return resources


def _build_exact_v3_database(path: Path) -> None:
    """Build the exact packaged v3 schema and durable migration history."""

    resources = _workflow_resources()
    hashes = {
        version: hashlib.sha256(resources[version - 1].read_bytes()).hexdigest()
        for version in (1, 2, 3)
    }
    applied_at = _STARTED_AT.isoformat()
    with connect_database(path) as connection:
        connection.executescript(resources[0].read_text(encoding="utf-8"))
        connection.execute(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (1, ?)",
            (applied_at,),
        )
        connection.commit()
        connection.executescript(resources[1].read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (2, ?)",
            (applied_at,),
        )
        connection.commit()
        db_core._install_legacy_archive_schema(connection)
        connection.commit()
        connection.executescript(resources[2].read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (3, ?)",
            (applied_at,),
        )
        connection.executemany(
            "INSERT INTO schema_migration_history("
            "ordinal, version, migration_sha256, applied_at) VALUES (?, ?, ?, ?)",
            [(version, version, hashes[version], applied_at) for version in (1, 2, 3)],
        )
        connection.commit()


def _seed_recoverable_case(
    path: Path,
    *,
    case_id: str,
    execution_state: str,
    execution_generation: int,
    with_previous_result: bool,
    prompt_tokens: int = 0,
) -> None:
    """Persist one exact input row accepted by aa9e43b's recovery selector."""

    source_id = f"src_{case_id}_snow_雪"
    previous: CaseResult | None = None
    if with_previous_result:
        previous = CaseResult(
            case_id=case_id,
            source_id=source_id,
            status=CaseStatus.INCOMPLETE,
            stop_reason="PREVIOUS_ATTEMPT_INCOMPLETE",
            errors=[
                ErrorRecord(
                    category="MODEL",
                    message="previous failure preserved verbatim: café 雪",
                    case_id=case_id,
                    stop_reason="PREVIOUS_FAILURE",
                    details={
                        "attempt": execution_generation,
                        "huge": 2**80 + 17,
                        "sqlite_json_values": [True, None, {"unicode": "雪"}],
                    },
                )
            ],
            usage=UsageSummary(prompt_tokens=prompt_tokens),
            started_at=_STARTED_AT,
            finished_at=_PREVIOUS_FINISHED_AT,
        )
    execution_token = None if execution_state == "IDLE" else f"exec_{'ef' * 16}"
    lease_expires_at = None if execution_state == "IDLE" else _EXPIRED_LEASE.isoformat()
    with connect_database(path) as connection:
        connection.execute(
            "INSERT INTO source_artifacts("
            "source_id, canonical_path, source_hash, source_format, size_bytes, "
            "modified_at, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                f"/historical/{case_id}.txt",
                hashlib.sha256(case_id.encode()).hexdigest(),
                "txt",
                1,
                _STARTED_AT.isoformat(),
                "{}",
                _STARTED_AT.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO cases("
            "case_id, source_id, status, stop_reason, result_json, started_at, updated_at, "
            "finished_at, execution_token, execution_generation, execution_state, "
            "lease_expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                case_id,
                source_id,
                "INCOMPLETE",
                previous.stop_reason if previous is not None else "CREATED",
                previous.model_dump_json() if previous is not None else None,
                _STARTED_AT.isoformat(),
                _PREVIOUS_FINISHED_AT.isoformat()
                if previous is not None
                else _STARTED_AT.isoformat(),
                _PREVIOUS_FINISHED_AT.isoformat() if previous is not None else None,
                execution_token,
                execution_generation,
                execution_state,
                lease_expires_at,
            ),
        )
        connection.commit()


def _replay_aa9e43b_recovery(path: Path) -> list[str]:
    """Replay aa9e43b's exact IDLE/RUNNING selection and generation transition.

    UUID randomness is replaced by fixed 32-lowerhex suffixes; the persisted
    CaseResult construction, compare-and-swap update, and recovery event match
    the historical implementation.
    """

    recovered: list[str] = []
    with connect_database(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute(
            "SELECT case_id, source_id, status, stop_reason, result_json, started_at, "
            "execution_token, execution_generation, execution_state, lease_expires_at "
            "FROM cases WHERE execution_state IN ('IDLE', 'RUNNING') "
            "AND status IN ('INCOMPLETE', 'NEEDS_HUMAN') ORDER BY case_id"
        ).fetchall()
        for row in rows:
            lease = (
                datetime.fromisoformat(row["lease_expires_at"])
                if row["lease_expires_at"] is not None
                else None
            )
            if row["execution_state"] == "RUNNING" and (lease is None or lease > _RECOVERED_AT):
                continue
            started_at = datetime.fromisoformat(row["started_at"])
            previous = (
                CaseResult.model_validate_json(row["result_json"], strict=True)
                if row["result_json"] is not None
                else None
            )
            recovery_error = ErrorRecord(
                category=ErrorCategory.ORCHESTRATION,
                message="execution lease expired before a terminal result was recorded",
                case_id=row["case_id"],
                stop_reason="ORPHANED_EXECUTION",
                details={"abandoned_execution_generation": int(row["execution_generation"])},
            )
            if previous is None:
                result = CaseResult(
                    case_id=row["case_id"],
                    source_id=row["source_id"],
                    status=CaseStatus.INCOMPLETE,
                    stop_reason="ORPHANED_EXECUTION",
                    errors=[recovery_error],
                    started_at=started_at,
                    finished_at=_RECOVERED_AT,
                )
            else:
                result = previous.model_copy(
                    update={
                        "status": CaseStatus.INCOMPLETE,
                        "stop_reason": "ORPHANED_EXECUTION",
                        "errors": [*previous.errors, recovery_error],
                        "finished_at": _RECOVERED_AT,
                    },
                    deep=True,
                )
            recovery_generation = int(row["execution_generation"]) + 1
            suffix = _TOKEN_SUFFIXES[str(row["case_id"])]
            recovery_token = f"recovery_{suffix}"
            updated = connection.execute(
                "UPDATE cases SET status = ?, stop_reason = ?, result_json = ?, "
                "updated_at = ?, finished_at = ?, execution_token = ?, "
                "execution_generation = ?, execution_state = 'FINISHED', "
                "lease_expires_at = NULL WHERE case_id = ? AND status = ? "
                "AND execution_token IS ? AND execution_generation = ? "
                "AND execution_state = ? AND lease_expires_at IS ?",
                (
                    result.status,
                    result.stop_reason,
                    result.model_dump_json(),
                    _RECOVERED_AT.isoformat(),
                    result.finished_at.isoformat(),
                    recovery_token,
                    recovery_generation,
                    row["case_id"],
                    row["status"],
                    row["execution_token"],
                    row["execution_generation"],
                    row["execution_state"],
                    row["lease_expires_at"],
                ),
            )
            assert updated.rowcount == 1
            connection.execute(
                "INSERT INTO events(event_id, case_id, source_id, event_type, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    f"evt_round7_{row['case_id']}",
                    row["case_id"],
                    row["source_id"],
                    "case.execution_recovered",
                    json.dumps(
                        {
                            "status": str(result.status),
                            "stop_reason": result.stop_reason,
                            "abandoned_execution_generation": int(row["execution_generation"]),
                            "recovery_execution_generation": recovery_generation,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    _RECOVERED_AT.isoformat(),
                ),
            )
            recovered.append(str(row["case_id"]))
        connection.commit()
    return recovered


def _case_row(path: Path, case_id: str) -> dict[str, Any]:
    with connect_database(path, read_only=True) as connection:
        row = connection.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    assert row is not None
    return dict(row)


def _logical_database_state(path: Path) -> dict[str, Any]:
    with connect_database(path, read_only=True) as connection:
        return {
            "schema": [
                tuple(row)
                for row in connection.execute(
                    "SELECT type, name, tbl_name, sql FROM sqlite_schema "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name, tbl_name"
                ).fetchall()
            ],
            "versions": [
                tuple(row)
                for row in connection.execute(
                    "SELECT version, applied_at FROM schema_version ORDER BY version"
                ).fetchall()
            ],
            "history": [
                tuple(row)
                for row in connection.execute(
                    "SELECT ordinal, version, migration_sha256, applied_at "
                    "FROM schema_migration_history ORDER BY ordinal"
                ).fetchall()
            ],
            "cases": [
                tuple(row)
                for row in connection.execute("SELECT * FROM cases ORDER BY case_id").fetchall()
            ],
            "events": [
                tuple(row)
                for row in connection.execute("SELECT * FROM events ORDER BY event_id").fetchall()
            ],
        }


def _assert_public_migration_failure_is_atomic(path: Path) -> None:
    before = _logical_database_state(path)
    with pytest.raises(DatabaseVerificationError) as excinfo:
        migrate_database(path, DatabaseKind.WORKFLOW)
    error = excinfo.value
    assert error.category is ErrorCategory.DATABASE
    assert error.stop_reason == "MIGRATION_FAILED"
    assert error.message == "database migration failed"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert _logical_database_state(path) == before


def _complete_strict_nested_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill every optional nested model with a coherent strict JSON value."""

    case_id = str(payload["case_id"])
    source_id = str(payload["source_id"])
    source = SourceArtifact(
        source_id=source_id,
        canonical_path=Path(f"/historical/{case_id}.txt"),
        sha256=hashlib.sha256(case_id.encode()).hexdigest(),
        source_format="txt",
        size_bytes=1,
        modified_at=_STARTED_AT,
    )
    critique = Critique(
        supported_findings=["Unicode evidence: café 雪"],
        challenged_findings=[],
        missing_evidence=[],
        requested_follow_up=[],
        recommended_disposition=DecisionKind.HOLD,
        rationale=["strict nested migration fixture"],
    )
    review = ReviewRequest(
        review_id="review_round7_nested",
        case_id=case_id,
        status="PENDING",
        reasons=["manual verification required"],
        amount=Money(amount=Decimal("12.34"), currency="USD"),
        source=source,
        evidence_bundle={
            "boolean": True,
            "null": None,
            "large": 2**80 + 17,
            "unicode": "雪",
        },
        agent_recommendation=DecisionKind.HOLD,
        agent_rationale=["hold until reviewed"],
        critic=critique,
        critic_disagreement_reason=None,
        questions=["Is the evidence complete?"],
        created_at=_PREVIOUS_FINISHED_AT,
    )
    decision = FinalDecision(
        decision=DecisionKind.HOLD,
        reasons=["review remains pending"],
        critic_disposition=DecisionKind.HOLD,
        payment_eligible=False,
    )
    payment = PaymentResult(
        payment_id=None,
        case_id=case_id,
        idempotency_key="nested-round7-not-eligible",
        status=PaymentStatus.NOT_ELIGIBLE,
        vendor=None,
        amount=None,
        processed_at=None,
    )
    completed = dict(payload)
    completed["final_decision"] = decision.model_dump(mode="json")
    completed["review_request"] = review.model_dump(mode="json")
    completed["payment"] = payment.model_dump(mode="json")
    return completed


@pytest.mark.parametrize(
    ("case_id", "execution_state", "abandoned_generation", "with_previous_result"),
    (
        ("case_round7_idle_zero", "IDLE", 0, False),
        ("case_round7_running", "RUNNING", 7, True),
    ),
)
def test_round7_public_v3_migration_replays_each_aa9e43b_recovery_path(
    tmp_path: Path,
    case_id: str,
    execution_state: str,
    abandoned_generation: int,
    with_previous_result: bool,
) -> None:
    """Both historical selector branches rekey after their exact generation advance."""

    path = tmp_path / f"workflow-v3-{execution_state.casefold()}.db"
    _build_exact_v3_database(path)
    _seed_recoverable_case(
        path,
        case_id=case_id,
        execution_state=execution_state,
        execution_generation=abandoned_generation,
        with_previous_result=with_previous_result,
    )
    assert _replay_aa9e43b_recovery(path) == [case_id]
    before = _case_row(path, case_id)
    parsed = CaseResult.model_validate_json(before["result_json"], strict=True)
    assert parsed.errors[-1].details == {"abandoned_execution_generation": abandoned_generation}
    assert before["execution_generation"] == abandoned_generation + 1
    assert before["execution_token"] == f"recovery_{_TOKEN_SUFFIXES[case_id]}"

    assert migrate_database(path, DatabaseKind.WORKFLOW) == [4, 5]

    after = _case_row(path, case_id)
    assert after == {
        **before,
        "execution_token": f"exec_{_TOKEN_SUFFIXES[case_id]}",
    }


def test_round7_public_migration_accepts_aa9e_wire_unicode_and_large_integer_values(
    tmp_path: Path,
) -> None:
    """Strict validation is value-based, not SQLite-int or exact-JSON-string based."""

    path = tmp_path / "workflow-v3-large-wire-values.db"
    case_id = "case_round7_large_usage"
    _build_exact_v3_database(path)
    _seed_recoverable_case(
        path,
        case_id=case_id,
        execution_state="RUNNING",
        execution_generation=3,
        with_previous_result=True,
        prompt_tokens=2**80 + 17,
    )
    assert _replay_aa9e43b_recovery(path) == [case_id]
    before = _case_row(path, case_id)
    assert "café 雪" in before["result_json"]
    parsed = CaseResult.model_validate_json(before["result_json"], strict=True)
    assert parsed.usage.prompt_tokens == 2**80 + 17
    assert parsed.final_decision is None
    assert parsed.review_request is None
    assert parsed.payment is None

    semantic_payload = json.loads(before["result_json"])
    equivalent_json = json.dumps(
        semantic_payload,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    assert equivalent_json != before["result_json"]
    assert "\\u96ea" in equivalent_json
    assert CaseResult.model_validate_json(equivalent_json, strict=True) == parsed
    with connect_database(path) as connection:
        connection.execute(
            "UPDATE cases SET result_json = ? WHERE case_id = ?",
            (equivalent_json, case_id),
        )
        connection.commit()
    before = _case_row(path, case_id)

    assert migrate_database(path, DatabaseKind.WORKFLOW) == [4, 5]
    assert _case_row(path, case_id) == {
        **before,
        "execution_token": f"exec_{_TOKEN_SUFFIXES[case_id]}",
    }


def test_round7_public_migration_accepts_complete_strict_nested_case_result(
    tmp_path: Path,
) -> None:
    """Valid decision, review/source, payment, usage, and error models all survive."""

    path = tmp_path / "workflow-v3-complete-nested-result.db"
    case_id = "case_round7_forged"
    _build_exact_v3_database(path)
    _seed_recoverable_case(
        path,
        case_id=case_id,
        execution_state="RUNNING",
        execution_generation=1,
        with_previous_result=False,
    )
    assert _replay_aa9e43b_recovery(path) == [case_id]
    before = _case_row(path, case_id)
    completed = _complete_strict_nested_payload(json.loads(before["result_json"]))
    encoded = json.dumps(completed, ensure_ascii=False, separators=(",", ":"))
    parsed = CaseResult.model_validate_json(encoded, strict=True)
    assert parsed.final_decision is not None
    assert parsed.review_request is not None
    assert parsed.payment is not None
    with connect_database(path) as connection:
        connection.execute(
            "UPDATE cases SET result_json = ? WHERE case_id = ?",
            (encoded, case_id),
        )
        connection.commit()
    before = _case_row(path, case_id)

    assert migrate_database(path, DatabaseKind.WORKFLOW) == [4, 5]
    assert _case_row(path, case_id) == {
        **before,
        "execution_token": f"exec_{_TOKEN_SUFFIXES[case_id]}",
    }


@pytest.mark.parametrize(
    "forgery",
    ("final_decision", "review_request", "payment", "usage", "review_source"),
)
def test_round7_public_migration_rejects_semantic_nested_model_corruption_atomically(
    tmp_path: Path,
    forgery: str,
) -> None:
    """Every rich nested model remains governed by its real strict constraints."""

    path = tmp_path / f"workflow-v3-semantic-forgery-{forgery}.db"
    case_id = "case_round7_forged"
    _build_exact_v3_database(path)
    _seed_recoverable_case(
        path,
        case_id=case_id,
        execution_state="RUNNING",
        execution_generation=1,
        with_previous_result=False,
    )
    assert _replay_aa9e43b_recovery(path) == [case_id]
    payload = _complete_strict_nested_payload(json.loads(_case_row(path, case_id)["result_json"]))
    if forgery == "final_decision":
        payload["final_decision"]["payment_eligible"] = True
    elif forgery == "review_request":
        payload["review_request"]["sequence"] = 0
    elif forgery == "payment":
        payload["payment"]["status"] = "FORGED"
    elif forgery == "usage":
        payload["usage"]["model_calls"] = False
    else:
        payload["review_request"]["source"]["size_bytes"] = -1
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with pytest.raises(ValidationError):
        CaseResult.model_validate_json(encoded, strict=True)
    with connect_database(path) as connection:
        connection.execute(
            "UPDATE cases SET result_json = ? WHERE case_id = ?",
            (encoded, case_id),
        )
        connection.commit()

    _assert_public_migration_failure_is_atomic(path)


@pytest.mark.parametrize(
    "forgery",
    ("final_decision", "review_request", "payment", "usage", "source_id"),
)
def test_round7_public_migration_rejects_each_strict_nested_case_result_forgery_atomically(
    tmp_path: Path,
    forgery: str,
) -> None:
    """An object-shaped nested payload is not evidence that its strict schema is valid."""

    path = tmp_path / f"workflow-v3-forged-{forgery}.db"
    _build_exact_v3_database(path)
    _seed_recoverable_case(
        path,
        case_id="case_round7_forged",
        execution_state="RUNNING",
        execution_generation=1,
        with_previous_result=False,
    )
    assert _replay_aa9e43b_recovery(path) == ["case_round7_forged"]
    payload = json.loads(_case_row(path, "case_round7_forged")["result_json"])
    if forgery == "usage":
        payload["usage"]["prompt_tokens"] = "0"
    else:
        payload[forgery] = {}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with pytest.raises(ValidationError):
        CaseResult.model_validate_json(encoded, strict=True)
    with connect_database(path) as connection:
        connection.execute(
            "UPDATE cases SET result_json = ? WHERE case_id = ?",
            (encoded, "case_round7_forged"),
        )
        connection.commit()

    _assert_public_migration_failure_is_atomic(path)


def test_round7_migration_sql_fails_closed_without_registered_strict_validator(
    tmp_path: Path,
) -> None:
    """Executing 004 without its deterministic UDF aborts before durable changes."""

    path = tmp_path / "workflow-v3-missing-validator.db"
    _build_exact_v3_database(path)
    _seed_recoverable_case(
        path,
        case_id="case_round7_forged",
        execution_state="RUNNING",
        execution_generation=1,
        with_previous_result=False,
    )
    assert _replay_aa9e43b_recovery(path) == ["case_round7_forged"]
    before = _logical_database_state(path)
    statements = db_core._migration_statements(_workflow_resources()[3].read_text(encoding="utf-8"))

    connection = sqlite3.connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="no such function"):
            for statement in statements:
                connection.execute(statement)
        connection.rollback()
    finally:
        connection.close()

    assert _logical_database_state(path) == before


@pytest.mark.parametrize("malformed_earlier_error", (42, {}, {"category": "MODEL"}))
def test_round7_public_migration_rejects_malformed_earlier_errors_atomically(
    tmp_path: Path,
    malformed_earlier_error: object,
) -> None:
    """A valid final recovery error cannot hide an invalid earlier error entry."""

    path = tmp_path / "workflow-v3-forged-earlier-error.db"
    _build_exact_v3_database(path)
    _seed_recoverable_case(
        path,
        case_id="case_round7_forged",
        execution_state="RUNNING",
        execution_generation=1,
        with_previous_result=False,
    )
    assert _replay_aa9e43b_recovery(path) == ["case_round7_forged"]
    payload = json.loads(_case_row(path, "case_round7_forged")["result_json"])
    payload["errors"].insert(0, malformed_earlier_error)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with pytest.raises(ValidationError):
        CaseResult.model_validate_json(encoded, strict=True)
    with connect_database(path) as connection:
        connection.execute(
            "UPDATE cases SET result_json = ? WHERE case_id = ?",
            (encoded, "case_round7_forged"),
        )
        connection.commit()

    _assert_public_migration_failure_is_atomic(path)
