"""Round 6 migration regressions for historical execution recovery authority."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import invoice_agents.db.core as db_core
from invoice_agents.db.core import (
    DatabaseKind,
    connect_database,
    migrate_database,
)
from invoice_agents.errors import (
    DatabaseVerificationError,
    ErrorCategory,
)

_CASE_ID = "case_round6_historical_recovery"
_SOURCE_ID = "src_round6_historical_recovery"
_STARTED_AT = "2026-08-09T12:00:00+00:00"
_RECOVERED_AT = "2026-08-09T12:05:00+00:00"
_RESULT_STARTED_AT = "2026-08-09T12:00:00Z"
_RESULT_RECOVERED_AT = "2026-08-09T12:05:00Z"
_LEGACY_TOKEN = f"recovery_{'1a' * 16}"
_CANONICAL_TOKEN = f"exec_{'1a' * 16}"
_LEASE = "2026-08-09T13:00:00+00:00"


def _historical_result() -> dict[str, Any]:
    return {
        "case_id": _CASE_ID,
        "source_id": _SOURCE_ID,
        "status": "INCOMPLETE",
        "stop_reason": "ORPHANED_EXECUTION",
        "final_decision": None,
        "review_request": None,
        "payment": None,
        "errors": [
            {
                "category": "ORCHESTRATION",
                "message": "execution lease expired before a terminal result was recorded",
                "case_id": _CASE_ID,
                "stop_reason": "ORPHANED_EXECUTION",
                "provider_request_id": None,
                "details": {"abandoned_execution_generation": 1},
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model_calls": 0,
            "tool_calls": 0,
            "retries": 0,
            "latency_ms": 0,
        },
        "started_at": _RESULT_STARTED_AT,
        "finished_at": _RESULT_RECOVERED_AT,
    }


def _workflow_resources() -> list[Any]:
    resources = db_core._migration_resources(DatabaseKind.WORKFLOW)
    assert [resource.name for resource in resources] == [
        "001_initial.sql",
        "002_review_sequence.sql",
        "003_execution_fencing.sql",
        "004_execution_token_grammar.sql",
    ]
    return resources


def _build_exact_v3_recovery_database(path: Path) -> None:
    """Build the exact packaged v3 schema/history plus one aa9e43b recovery row."""

    resources = _workflow_resources()
    hashes = {
        version: hashlib.sha256(resources[version - 1].read_bytes()).hexdigest()
        for version in (1, 2, 3)
    }
    with connect_database(path) as connection:
        connection.executescript(resources[0].read_text(encoding="utf-8"))
        connection.execute(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (1, ?)",
            (_STARTED_AT,),
        )
        connection.commit()

        connection.executescript(resources[1].read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (2, ?)",
            (_STARTED_AT,),
        )
        connection.commit()

        db_core._install_legacy_archive_schema(connection)
        connection.commit()
        connection.executescript(resources[2].read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (3, ?)",
            (_STARTED_AT,),
        )
        connection.executemany(
            "INSERT INTO schema_migration_history("
            "ordinal, version, migration_sha256, applied_at) VALUES (?, ?, ?, ?)",
            [(version, version, hashes[version], _STARTED_AT) for version in (1, 2, 3)],
        )
        connection.execute(
            "INSERT INTO source_artifacts("
            "source_id, canonical_path, source_hash, source_format, size_bytes, "
            "modified_at, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _SOURCE_ID,
                "/historical/round6.txt",
                "a" * 64,
                "txt",
                1,
                _STARTED_AT,
                "{}",
                _STARTED_AT,
            ),
        )
        connection.execute(
            "INSERT INTO cases("
            "case_id, source_id, status, stop_reason, result_json, started_at, updated_at, "
            "finished_at, execution_token, execution_generation, execution_state, "
            "lease_expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _CASE_ID,
                _SOURCE_ID,
                "INCOMPLETE",
                "ORPHANED_EXECUTION",
                json.dumps(_historical_result(), separators=(",", ":")),
                _STARTED_AT,
                _RECOVERED_AT,
                _RECOVERED_AT,
                _LEGACY_TOKEN,
                2,
                "FINISHED",
                None,
            ),
        )
        connection.commit()


def _case_row(path: Path) -> dict[str, Any]:
    with connect_database(path, read_only=True) as connection:
        row = connection.execute(
            "SELECT * FROM cases WHERE case_id = ?",
            (_CASE_ID,),
        ).fetchone()
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
            "case": _case_row(path),
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


def _replace_result(path: Path, mutate: str) -> None:
    result = _historical_result()
    if mutate == "wrong_result_status":
        result["status"] = "FAILED"
    elif mutate == "result_source_mismatch":
        result["source_id"] = "src_other"
    elif mutate == "wrong_recovery_error":
        result["errors"][-1]["message"] = "different recovery failure"
    elif mutate == "wrong_recovery_generation_link":
        result["errors"][-1]["details"]["abandoned_execution_generation"] = 2
    elif mutate == "extra_result_field":
        result["legacy_extra"] = True
    else:  # pragma: no cover - the test table is the complete caller contract
        raise AssertionError(f"unknown result mutation: {mutate}")
    with connect_database(path) as connection:
        connection.execute(
            "UPDATE cases SET result_json = ? WHERE case_id = ?",
            (json.dumps(result, separators=(",", ":")), _CASE_ID),
        )
        connection.commit()


def _update_authority_without_leaving_schema_damage(
    path: Path,
    statement: str,
    parameters: tuple[Any, ...],
) -> None:
    with connect_database(path) as connection:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type = 'trigger' "
            "AND name = 'trg_cases_execution_authority_update'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER trg_cases_execution_authority_update")
        connection.execute(statement, parameters)
        connection.execute(trigger_sql)
        connection.commit()


def _apply_contradiction(path: Path, contradiction: str) -> None:
    if contradiction in {
        "wrong_result_status",
        "result_source_mismatch",
        "wrong_recovery_error",
        "wrong_recovery_generation_link",
        "extra_result_field",
    }:
        _replace_result(path, contradiction)
        return
    if contradiction == "finished_with_lease":
        _update_authority_without_leaving_schema_damage(
            path,
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            (_LEASE, _CASE_ID),
        )
        return
    if contradiction in {"generation_real", "generation_text"}:
        invalid_generation: object = 2.5 if contradiction == "generation_real" else "two"
        _update_authority_without_leaving_schema_damage(
            path,
            "UPDATE cases SET execution_generation = ? WHERE case_id = ?",
            (invalid_generation, _CASE_ID),
        )
        return

    statements: dict[str, tuple[str, tuple[Any, ...]]] = {
        "running_recovery": (
            "UPDATE cases SET execution_state = 'RUNNING', lease_expires_at = ? WHERE case_id = ?",
            (_LEASE, _CASE_ID),
        ),
        "wrong_case_status": (
            "UPDATE cases SET status = 'FAILED' WHERE case_id = ?",
            (_CASE_ID,),
        ),
        "wrong_case_stop_reason": (
            "UPDATE cases SET stop_reason = 'CANCELLED' WHERE case_id = ?",
            (_CASE_ID,),
        ),
        "missing_result": (
            "UPDATE cases SET result_json = NULL WHERE case_id = ?",
            (_CASE_ID,),
        ),
        "generation_one": (
            "UPDATE cases SET execution_generation = 1 WHERE case_id = ?",
            (_CASE_ID,),
        ),
    }
    statement, parameters = statements[contradiction]
    with connect_database(path) as connection:
        connection.execute(statement, parameters)
        connection.commit()


def test_round6_public_v3_migration_reconciles_exact_historical_recovery_token(
    tmp_path: Path,
) -> None:
    """The exact aa9e43b terminal recovery row upgrades through the public worker."""

    path = tmp_path / "workflow-v3-historical-recovery.db"
    _build_exact_v3_recovery_database(path)
    before = _case_row(path)
    assert before["execution_token"] == _LEGACY_TOKEN

    assert migrate_database(path, DatabaseKind.WORKFLOW) == [4]

    after = _case_row(path)
    assert after == {**before, "execution_token": _CANONICAL_TOKEN}
    resources = _workflow_resources()
    expected_v4_hash = hashlib.sha256(resources[3].read_bytes()).hexdigest()
    with connect_database(path, read_only=True) as connection:
        assert [
            row[0]
            for row in connection.execute(
                "SELECT version FROM schema_version ORDER BY version"
            ).fetchall()
        ] == [1, 2, 3, 4]
        history_row = connection.execute(
            "SELECT ordinal, migration_sha256 FROM schema_migration_history WHERE version = 4"
        ).fetchone()
        assert history_row is not None
        assert tuple(history_row) == (4, expected_v4_hash)
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name IN ("
                "'execution_token_migration_guard', "
                "'execution_token_recovery_reconciliation')"
            ).fetchone()[0]
            == 0
        )
    assert migrate_database(path, DatabaseKind.WORKFLOW) == []


@pytest.mark.parametrize(
    "contradiction",
    (
        "running_recovery",
        "wrong_case_status",
        "wrong_case_stop_reason",
        "missing_result",
        "wrong_result_status",
        "result_source_mismatch",
        "wrong_recovery_error",
        "wrong_recovery_generation_link",
        "extra_result_field",
        "generation_one",
        "generation_real",
        "generation_text",
        "finished_with_lease",
    ),
)
def test_round6_public_v3_migration_rejects_contradictory_recovery_atomically(
    tmp_path: Path,
    contradiction: str,
) -> None:
    """Each non-historical shape fails publicly without changing v3 schema/history/data."""

    path = tmp_path / f"workflow-v3-contradictory-{contradiction}.db"
    _build_exact_v3_recovery_database(path)
    _apply_contradiction(path, contradiction)

    _assert_public_migration_failure_is_atomic(path)


_RECOVERY_TOKEN_NEAR_MATCHES = (
    ("uppercase_prefix", f"Recovery_{'a' * 32}"),
    ("uppercase_hex", f"recovery_{'A' * 32}"),
    ("short", f"recovery_{'a' * 31}"),
    ("long", f"recovery_{'a' * 33}"),
    ("nonhex", f"recovery_{'g' * 32}"),
    ("leading_whitespace", f" recovery_{'a' * 32}"),
    ("trailing_whitespace", f"recovery_{'a' * 32} "),
    ("unicode", f"recovery_{'a' * 31}é"),
    ("nul", f"recovery_{'a' * 31}\x00"),
    ("wrong_separator", f"recovery-{'a' * 32}"),
    ("near_prefix", f"recoverx_{'a' * 32}"),
)


@pytest.mark.parametrize(
    ("label", "malformed_token"),
    _RECOVERY_TOKEN_NEAR_MATCHES,
    ids=[case[0] for case in _RECOVERY_TOKEN_NEAR_MATCHES],
)
def test_round6_public_v3_migration_rejects_recovery_token_near_matches_atomically(
    tmp_path: Path,
    label: str,
    malformed_token: str,
) -> None:
    """Only recovery_ followed by exactly 32 lowercase hex characters is eligible."""

    path = tmp_path / f"workflow-v3-token-{label}.db"
    _build_exact_v3_recovery_database(path)
    with connect_database(path) as connection:
        connection.execute(
            "UPDATE cases SET execution_token = ? WHERE case_id = ?",
            (malformed_token, _CASE_ID),
        )
        connection.commit()

    _assert_public_migration_failure_is_atomic(path)


def test_round6_v4_insert_and_update_triggers_enforce_exact_exec_token_grammar(
    tmp_path: Path,
) -> None:
    """Both direct-SQL entry paths reject every near match and accept canonical exec tokens."""

    path = tmp_path / "workflow-v4-trigger-grammar.db"
    _build_exact_v3_recovery_database(path)
    assert migrate_database(path, DatabaseKind.WORKFLOW) == [4]
    malformed_tokens = (
        "exec_short",
        f"exec_{'a' * 31}",
        f"exec_{'a' * 33}",
        f"exec_{'A' * 32}",
        f"exec_{'g' * 32}",
        f"exec_{'a' * 31}é",
        f"exec_{'a' * 31}\x00",
        f"exec_{'a' * 32} ",
        f"recovery_{'a' * 32}",
    )
    with connect_database(path) as connection:
        connection.execute(
            "INSERT INTO cases(case_id, status, started_at, updated_at) "
            "VALUES ('case_round6_update', 'INCOMPLETE', ?, ?)",
            (_STARTED_AT, _STARTED_AT),
        )
        for index, malformed_token in enumerate(malformed_tokens):
            with pytest.raises(sqlite3.IntegrityError, match="INVALID_EXECUTION_TOKEN"):
                connection.execute(
                    "INSERT INTO cases("
                    "case_id, status, started_at, updated_at, execution_token, "
                    "execution_generation, execution_state, lease_expires_at) "
                    "VALUES (?, 'INCOMPLETE', ?, ?, ?, 1, 'RUNNING', ?)",
                    (
                        f"case_round6_insert_{index}",
                        _STARTED_AT,
                        _STARTED_AT,
                        malformed_token,
                        _LEASE,
                    ),
                )
            with pytest.raises(sqlite3.IntegrityError, match="INVALID_EXECUTION_TOKEN"):
                connection.execute(
                    "UPDATE cases SET execution_token = ?, execution_generation = 1, "
                    "execution_state = 'FINISHED', lease_expires_at = NULL "
                    "WHERE case_id = 'case_round6_update'",
                    (malformed_token,),
                )

        insert_token = f"exec_{'0f' * 16}"
        update_token = f"exec_{'f0' * 16}"
        connection.execute(
            "INSERT INTO cases("
            "case_id, status, started_at, updated_at, execution_token, "
            "execution_generation, execution_state, lease_expires_at) "
            "VALUES ('case_round6_insert_valid', 'INCOMPLETE', ?, ?, ?, 1, "
            "'RUNNING', ?)",
            (_STARTED_AT, _STARTED_AT, insert_token, _LEASE),
        )
        connection.execute(
            "UPDATE cases SET execution_token = ?, execution_generation = 1, "
            "execution_state = 'FINISHED', lease_expires_at = NULL "
            "WHERE case_id = 'case_round6_update'",
            (update_token,),
        )
        connection.commit()
        assert (
            connection.execute(
                "SELECT execution_token FROM cases WHERE case_id = 'case_round6_insert_valid'"
            ).fetchone()[0]
            == insert_token
        )
        assert (
            connection.execute(
                "SELECT execution_token FROM cases WHERE case_id = 'case_round6_update'"
            ).fetchone()[0]
            == update_token
        )
