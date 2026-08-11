"""Forward-v5 schema and store contracts for durable result-artifact bindings."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import stat
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import cast

import pytest

import invoice_agents.db.core as db_core
import invoice_agents.db.store as store_module
from invoice_agents.config import Settings
from invoice_agents.db.core import (
    DatabaseKind,
    connect_database,
    verify_database,
)
from invoice_agents.db.store import (
    ExecutionClaim,
    ResultArtifactBinding,
    WorkflowStore,
)
from invoice_agents.errors import DatabaseVerificationError, InvoiceAgentsError
from invoice_agents.models import CaseResult, CaseStatus
from invoice_agents.source_store import snapshot_source

MIGRATION_001_SHA256 = "b4e3f58c36aec8dfa3a41b780cca7337345e756351f264f8d759cfd7b47c0bd7"
MIGRATION_002_SHA256 = "b16614a8699c40074ff30531251126e964ed2eb86a2afaf53533f367447beb02"
MIGRATION_003_SHA256 = "8950c2fd5fca058ea2256b7fab52ffceee3389e92b8654bff626acf75313e5aa"
MIGRATION_004_SHA256 = "0f46037fd9300cd584ba88fcbd3084f254689d2546191be941299037cc39f5d2"
# Replace this sentinel exactly once, after migration 005 has passed review and
# its bytes are frozen.  Computing the expectation from the resource would let
# a migration rewrite bless itself.
MIGRATION_005_SHA256 = "96ac6ca33808f0ee9458aa74757ebc437c70b5f1178e6cacd9490b882855db1c"
MIGRATION_005_NAME = "005_result_artifact_bindings.sql"
STARTED_AT = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
FINISHED_AT = STARTED_AT + timedelta(seconds=1)
SOURCE_PATH = Path(__file__).resolve().parents[2] / "data/invoices/invoice_1001.txt"
PARENT_TRIGGER_ERROR = "RESULT_ARTIFACT_BINDING_PARENT_INVALID"
BINDING_TRIGGER_NAMES = {
    "trg_result_artifact_bindings_terminal_insert",
    "trg_result_artifact_bindings_terminal_update",
    "trg_cases_result_artifact_binding_guard_update",
}
EXECUTION_TOKEN_GLOB = "exec_" + "[0-9a-f]" * 32
EXPECTED_BINDING_TABLE_SQL = """
CREATE TABLE result_artifact_bindings (
    case_id TEXT NOT NULL PRIMARY KEY,
    execution_generation INTEGER NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    artifact_device INTEGER NOT NULL,
    artifact_inode INTEGER NOT NULL,
    artifact_file_type INTEGER NOT NULL,
    artifact_size_bytes INTEGER NOT NULL,
    FOREIGN KEY(case_id, execution_generation)
        REFERENCES cases(case_id, execution_generation)
        ON UPDATE RESTRICT ON DELETE CASCADE,
    CHECK(typeof(execution_generation) = 'integer' AND execution_generation >= 1),
    CHECK(
        typeof(artifact_sha256) = 'text'
        AND length(artifact_sha256) = 64
        AND artifact_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK(typeof(artifact_device) = 'integer' AND artifact_device >= 0),
    CHECK(typeof(artifact_inode) = 'integer' AND artifact_inode > 0),
    CHECK(typeof(artifact_file_type) = 'integer' AND artifact_file_type = 32768),
    CHECK(typeof(artifact_size_bytes) = 'integer' AND artifact_size_bytes >= 0)
)
"""
EXPECTED_PARENT_INDEX_SQL = """
CREATE UNIQUE INDEX idx_cases_case_generation
ON cases(case_id, execution_generation)
"""
EXPECTED_BINDING_TRIGGER_SQL = {
    "trg_result_artifact_bindings_terminal_insert": f"""
        CREATE TRIGGER trg_result_artifact_bindings_terminal_insert
        BEFORE INSERT ON result_artifact_bindings
        WHEN NOT EXISTS (
            SELECT 1 FROM cases AS parent
            WHERE parent.case_id = NEW.case_id
            AND parent.execution_generation = NEW.execution_generation
            AND parent.execution_state = 'FINISHED'
            AND typeof(parent.execution_token) = 'text'
            AND length(parent.execution_token) = 37
            AND parent.execution_token GLOB '{EXECUTION_TOKEN_GLOB}'
            AND typeof(parent.result_json) = 'text'
            AND parent.lease_expires_at IS NULL
        )
        BEGIN
            SELECT RAISE(ABORT, 'RESULT_ARTIFACT_BINDING_PARENT_INVALID');
        END
    """,
    "trg_result_artifact_bindings_terminal_update": f"""
        CREATE TRIGGER trg_result_artifact_bindings_terminal_update
        BEFORE UPDATE ON result_artifact_bindings
        WHEN NOT EXISTS (
            SELECT 1 FROM cases AS parent
            WHERE parent.case_id = NEW.case_id
            AND parent.execution_generation = NEW.execution_generation
            AND parent.execution_state = 'FINISHED'
            AND typeof(parent.execution_token) = 'text'
            AND length(parent.execution_token) = 37
            AND parent.execution_token GLOB '{EXECUTION_TOKEN_GLOB}'
            AND typeof(parent.result_json) = 'text'
            AND parent.lease_expires_at IS NULL
        )
        BEGIN
            SELECT RAISE(ABORT, 'RESULT_ARTIFACT_BINDING_PARENT_INVALID');
        END
    """,
    "trg_cases_result_artifact_binding_guard_update": """
        CREATE TRIGGER trg_cases_result_artifact_binding_guard_update
        BEFORE UPDATE OF execution_token, execution_generation, execution_state,
            lease_expires_at, result_json
        ON cases
        WHEN EXISTS (
            SELECT 1 FROM result_artifact_bindings AS binding
            WHERE binding.case_id = OLD.case_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'RESULT_ARTIFACT_BINDING_MUST_BE_INVALIDATED');
        END
    """,
}
MIGRATION_SHA256 = {
    1: MIGRATION_001_SHA256,
    2: MIGRATION_002_SHA256,
    3: MIGRATION_003_SHA256,
    4: MIGRATION_004_SHA256,
    5: MIGRATION_005_SHA256,
}
_BINDING_KERNEL_SEAMS = {
    "execute": "_execute_result_artifact_binding",
    "commit": "_commit_result_artifact_binding",
    "rollback": "_rollback_result_artifact_binding",
}


def _workflow_resources() -> list[Traversable]:
    return list(db_core._migration_resources(DatabaseKind.WORKFLOW))


def _resource_version(resource: Traversable) -> int:
    return int(str(resource.name).split("_", 1)[0])


def _migration_resource(version: int) -> Traversable:
    matches = [resource for resource in _workflow_resources() if _resource_version(resource) == version]
    assert len(matches) == 1
    return matches[0]


def _binding(case_id: str, generation: int, *, marker: int = 1) -> ResultArtifactBinding:
    return ResultArtifactBinding(
        case_id=case_id,
        execution_generation=generation,
        artifact_sha256=f"{marker:x}" * 64,
        artifact_device=marker,
        artifact_inode=marker,
        artifact_file_type=stat.S_IFREG,
        artifact_size_bytes=marker,
    )


def _new_case(settings: Settings, case_id: str) -> tuple[WorkflowStore, ExecutionClaim, str]:
    source = snapshot_source(SOURCE_PATH, settings.source_archive_dir, max_bytes=10_485_760)
    store = WorkflowStore(settings)
    store.register_source(source)
    store.create_case(case_id, source, STARTED_AT)
    claim = store.claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    return store, claim, source.source_id


def _terminal_result(case_id: str, source_id: str, *, suffix: str = "ONE") -> CaseResult:
    return CaseResult(
        case_id=case_id,
        source_id=source_id,
        status=CaseStatus.FAILED,
        stop_reason=f"BINDING_TEST_{suffix}",
        started_at=STARTED_AT,
        finished_at=FINISHED_AT if suffix == "ONE" else FINISHED_AT + timedelta(seconds=1),
    )


def _finished_case(
    settings: Settings,
    case_id: str,
    *,
    bind: bool,
) -> tuple[WorkflowStore, ExecutionClaim, CaseResult]:
    store, claim, source_id = _new_case(settings, case_id)
    result = _terminal_result(case_id, source_id)
    store.finish_case(result, claim)
    if bind:
        store.save_result_artifact_binding(_binding(case_id, claim.generation), result)
    return store, claim, result


def _trigger_definitions(connection: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    rows = connection.execute(
        "SELECT name, sql FROM sqlite_schema WHERE type = 'trigger' AND tbl_name = ? "
        "ORDER BY name",
        (table,),
    ).fetchall()
    definitions = [(str(row["name"]), str(row["sql"])) for row in rows]
    assert all(sql and sql != "None" for _name, sql in definitions)
    return definitions


def _without_table_triggers(
    connection: sqlite3.Connection,
    table: str,
    operation: Callable[[], object],
) -> None:
    definitions = _trigger_definitions(connection, table)
    for name, _sql in definitions:
        connection.execute(f'DROP TRIGGER "{name}"')
    operation()
    for _name, sql in definitions:
        connection.execute(sql)


def _corrupt_case(
    settings: Settings,
    case_id: str,
    assignments: str,
    parameters: tuple[object, ...],
) -> None:
    with connect_database(settings.workflow_db) as connection:
        _without_table_triggers(
            connection,
            "cases",
            lambda: connection.execute(
                f"UPDATE cases SET {assignments} WHERE case_id = ?",
                (*parameters, case_id),
            ),
        )
        connection.commit()


def _insert_raw_binding(
    connection: sqlite3.Connection,
    values: tuple[object, ...],
) -> None:
    assert len(values) == 7
    connection.execute(
        "INSERT INTO result_artifact_bindings("
        "case_id, execution_generation, artifact_sha256, artifact_device, "
        "artifact_inode, artifact_file_type, artifact_size_bytes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        values,
    )


def _binding_values(case_id: str, generation: int) -> tuple[object, ...]:
    return (case_id, generation, "a" * 64, 1, 1, stat.S_IFREG, 1)


def _inject_binding_without_binding_triggers(
    settings: Settings,
    case_id: str,
    generation: int,
) -> None:
    with connect_database(settings.workflow_db) as connection:
        _without_table_triggers(
            connection,
            "result_artifact_bindings",
            lambda: _insert_raw_binding(connection, _binding_values(case_id, generation)),
        )
        connection.commit()


def _patch_binding_kernel_action(
    monkeypatch: pytest.MonkeyPatch,
    *,
    action: str,
    timing: str,
    failure: BaseException,
) -> None:
    """Inject before/after a real SQLite action through the required narrow seam."""

    seam_name = _BINDING_KERNEL_SEAMS[action]
    if action == "execute":
        installed_execute = cast(
            Callable[[sqlite3.Connection, str, tuple[object, ...]], sqlite3.Cursor] | None,
            getattr(store_module, seam_name, None),
        )
        if installed_execute is None:

            def installed_execute(
                connection: sqlite3.Connection,
                statement: str,
                parameters: tuple[object, ...],
            ) -> sqlite3.Cursor:
                return connection.execute(statement, parameters)

        def inject_execute(
            connection: sqlite3.Connection,
            statement: str,
            parameters: tuple[object, ...],
        ) -> sqlite3.Cursor:
            if timing == "before":
                raise failure
            installed_execute(connection, statement, parameters)
            raise failure

        injected: object = inject_execute
    else:
        installed_boundary = cast(
            Callable[[sqlite3.Connection], None] | None,
            getattr(store_module, seam_name, None),
        )
        if installed_boundary is None:

            def installed_boundary(connection: sqlite3.Connection) -> None:
                if action == "commit":
                    connection.commit()
                else:
                    connection.rollback()

        def inject_boundary(connection: sqlite3.Connection) -> None:
            if timing == "before":
                raise failure
            installed_boundary(connection)
            raise failure

        injected = inject_boundary

    # raising=False intentionally makes an implementation with inline kernel
    # calls fail the behavioral assertion: adding an unused test hook is not
    # enough; the real binding operation must route through this seam.
    monkeypatch.setattr(store_module, seam_name, injected, raising=False)


def _assert_exact_failure(
    invocation: Callable[[], None],
    expected: BaseException,
) -> None:
    with pytest.raises(BaseException) as excinfo:
        invocation()
    assert excinfo.value is expected
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def _control_failure(kind: str, marker: str) -> BaseException:
    if kind == "keyboard-interrupt":
        return KeyboardInterrupt(marker)
    if kind == "system-exit":
        return SystemExit(marker)
    if kind == "cancelled-error":
        return asyncio.CancelledError(marker)
    raise AssertionError(f"unsupported control kind: {kind}")


def _make_v4_database(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resources = db_core._migration_resources
    v4_resources = [
        resource
        for resource in original_resources(DatabaseKind.WORKFLOW)
        if _resource_version(resource) <= 4
    ]

    def resources_through_v4(kind: DatabaseKind) -> list[Traversable]:
        if kind is DatabaseKind.WORKFLOW:
            return v4_resources
        return list(original_resources(kind))

    monkeypatch.setattr(db_core, "_migration_resources", resources_through_v4)
    db_core._expected_workflow_schema_manifest.cache_clear()
    assert db_core._migrate_database_in_process(path, DatabaseKind.WORKFLOW) == [1, 2, 3, 4]
    monkeypatch.setattr(db_core, "_migration_resources", original_resources)
    db_core._expected_workflow_schema_manifest.cache_clear()


def test_migration_004_is_immutable_and_migration_005_has_a_reviewed_literal_digest() -> None:
    for version in range(1, 4):
        migration = _migration_resource(version)
        assert hashlib.sha256(migration.read_bytes()).hexdigest() == MIGRATION_SHA256[version]

    migration_004 = _migration_resource(4)
    assert migration_004.name == "004_execution_token_grammar.sql"
    actual_004 = hashlib.sha256(migration_004.read_bytes()).hexdigest()
    assert actual_004 == MIGRATION_004_SHA256, (
        "migration 004 is immutable; restore its independently reviewed bytes instead of "
        f"blessing digest {actual_004}"
    )

    migration_005 = _migration_resource(5)
    assert migration_005.name == MIGRATION_005_NAME
    actual_005 = hashlib.sha256(migration_005.read_bytes()).hexdigest()
    assert actual_005 == MIGRATION_SHA256[5], (
        "freeze the reviewed migration 005 bytes and replace the literal test sentinel with "
        f"{actual_005}"
    )


def test_existing_v4_database_upgrades_to_v5_without_inventing_artifact_identity(
    tmp_path: Path,
    inventory_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "workflow-v4.db"
    _make_v4_database(path, monkeypatch)
    historical = CaseResult(
        case_id="case_v4_historical",
        source_id=None,
        status=CaseStatus.FAILED,
        stop_reason="V4_HISTORICAL_TERMINAL",
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )
    with connect_database(path) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE name = 'result_artifact_bindings'"
            ).fetchone()
            is None
        )
        connection.execute(
            "INSERT INTO cases(case_id, source_id, status, stop_reason, result_json, "
            "started_at, updated_at, finished_at, execution_token, execution_generation, "
            "execution_state, lease_expires_at) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, 1, "
            "'FINISHED', NULL)",
            (
                historical.case_id,
                historical.status,
                historical.stop_reason,
                historical.model_dump_json(),
                STARTED_AT.isoformat(),
                FINISHED_AT.isoformat(),
                FINISHED_AT.isoformat(),
                "exec_" + "a" * 32,
            ),
        )
        connection.commit()

    settings = Settings(
        workflow_db=path,
        inventory_db=inventory_db,
        source_archive_dir=tmp_path / "sources",
    )
    assert db_core._migrate_database_in_process(
        path,
        DatabaseKind.WORKFLOW,
        settings=settings,
    ) == [5]
    assert verify_database(path, DatabaseKind.WORKFLOW, settings=settings)["schema_version"] == 5
    with connect_database(path, read_only=True) as connection:
        assert [
            int(row["version"])
            for row in connection.execute("SELECT version FROM schema_version ORDER BY version")
        ] == [1, 2, 3, 4, 5]
        history = connection.execute(
            "SELECT version, migration_sha256 FROM schema_migration_history ORDER BY ordinal"
        ).fetchall()
        assert history[3]["migration_sha256"] == MIGRATION_004_SHA256
        assert history[4]["migration_sha256"] == MIGRATION_005_SHA256
        assert connection.execute("SELECT COUNT(*) FROM result_artifact_bindings").fetchone()[0] == 0
    assert WorkflowStore(settings).load_result("case_v4_historical") == historical


def test_v5_binding_schema_has_exact_columns_index_foreign_key_and_triggers(
    workflow_db: Path,
) -> None:
    with connect_database(workflow_db, read_only=True) as connection:
        version = connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        table_row = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type = 'table' "
            "AND name = 'result_artifact_bindings'"
        ).fetchone()
        columns = [
            tuple(row)
            for row in connection.execute(
                "PRAGMA table_info('result_artifact_bindings')"
            ).fetchall()
        ]
        foreign_keys = [
            tuple(row)
            for row in connection.execute(
                "PRAGMA foreign_key_list('result_artifact_bindings')"
            ).fetchall()
        ]
        parent_index = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type = 'index' "
            "AND name = 'idx_cases_case_generation'"
        ).fetchone()
        parent_index_columns = [
            str(row["name"])
            for row in connection.execute(
                "PRAGMA index_info('idx_cases_case_generation')"
            ).fetchall()
        ]
        trigger_sql = {
            str(row["name"]): str(row["sql"])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_schema WHERE type = 'trigger' "
                "AND (tbl_name = 'result_artifact_bindings' OR "
                "name = 'trg_cases_result_artifact_binding_guard_update')"
            ).fetchall()
        }

    assert version == 5
    assert table_row is not None
    assert db_core._normalized_sql(str(table_row["sql"])) == db_core._normalized_sql(
        EXPECTED_BINDING_TABLE_SQL
    )
    assert columns == [
        (0, "case_id", "TEXT", 1, None, 1),
        (1, "execution_generation", "INTEGER", 1, None, 0),
        (2, "artifact_sha256", "TEXT", 1, None, 0),
        (3, "artifact_device", "INTEGER", 1, None, 0),
        (4, "artifact_inode", "INTEGER", 1, None, 0),
        (5, "artifact_file_type", "INTEGER", 1, None, 0),
        (6, "artifact_size_bytes", "INTEGER", 1, None, 0),
    ]
    assert foreign_keys == [
        (0, 0, "cases", "case_id", "case_id", "RESTRICT", "CASCADE", "NONE"),
        (
            0,
            1,
            "cases",
            "execution_generation",
            "execution_generation",
            "RESTRICT",
            "CASCADE",
            "NONE",
        ),
    ]
    assert parent_index is not None
    assert db_core._normalized_sql(str(parent_index["sql"])) == db_core._normalized_sql(
        EXPECTED_PARENT_INDEX_SQL
    )
    assert parent_index_columns == ["case_id", "execution_generation"]
    assert set(trigger_sql) == BINDING_TRIGGER_NAMES
    assert {
        name: db_core._normalized_sql(sql) for name, sql in trigger_sql.items()
    } == {
        name: db_core._normalized_sql(sql)
        for name, sql in EXPECTED_BINDING_TRIGGER_SQL.items()
    }


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        pytest.param("case_id", None, id="null-case-id"),
        pytest.param("execution_generation", 0, id="zero-generation"),
        pytest.param("execution_generation", 1.5, id="noninteger-generation"),
        pytest.param("artifact_sha256", "a" * 63, id="short-sha"),
        pytest.param("artifact_sha256", "A" * 64, id="uppercase-sha"),
        pytest.param("artifact_sha256", "g" * 64, id="nonhex-sha"),
        pytest.param("artifact_device", -1, id="negative-device"),
        pytest.param("artifact_device", 1.5, id="noninteger-device"),
        pytest.param("artifact_inode", 0, id="zero-inode"),
        pytest.param("artifact_inode", 1.5, id="noninteger-inode"),
        pytest.param("artifact_file_type", 0, id="nonregular-file-type"),
        pytest.param(
            "artifact_file_type",
            stat.S_IFREG + 0.5,
            id="noninteger-file-type",
        ),
        pytest.param("artifact_size_bytes", -1, id="negative-size"),
        pytest.param("artifact_size_bytes", 1.5, id="noninteger-size"),
    ],
)
def test_binding_table_rejects_every_invalid_scalar(
    field: str,
    invalid: object,
    settings: Settings,
) -> None:
    store, claim, _result = _finished_case(settings, f"case_invalid_{field}", bind=False)
    del store
    names = [
        "case_id",
        "execution_generation",
        "artifact_sha256",
        "artifact_device",
        "artifact_inode",
        "artifact_file_type",
        "artifact_size_bytes",
    ]
    values = list(_binding_values(claim.case_id, claim.generation))
    values[names.index(field)] = invalid

    with connect_database(settings.workflow_db) as connection, pytest.raises(
        sqlite3.IntegrityError
    ):
        _insert_raw_binding(connection, tuple(values))


@pytest.mark.parametrize(
    ("case_suffix", "assignments", "parameters"),
    [
        pytest.param(
            "running",
            "execution_state = 'RUNNING', lease_expires_at = ?",
            ((FINISHED_AT + timedelta(minutes=5)).isoformat(),),
            id="running-predecessor-result",
        ),
        pytest.param("missing_result", "result_json = NULL", (), id="missing-result"),
        pytest.param("null_token", "execution_token = NULL", (), id="null-token"),
        pytest.param(
            "malformed_token",
            "execution_token = ?",
            ("exec_" + "g" * 32,),
            id="malformed-token",
        ),
        pytest.param(
            "finished_lease",
            "lease_expires_at = ?",
            ((FINISHED_AT + timedelta(minutes=5)).isoformat(),),
            id="finished-with-lease",
        ),
    ],
)
def test_binding_trigger_rejects_noncanonical_terminal_parent(
    case_suffix: str,
    assignments: str,
    parameters: tuple[object, ...],
    settings: Settings,
) -> None:
    case_id = f"case_parent_{case_suffix}"
    _store, claim, _result = _finished_case(settings, case_id, bind=False)
    _corrupt_case(settings, case_id, assignments, parameters)

    with connect_database(settings.workflow_db) as connection, pytest.raises(
        sqlite3.IntegrityError,
        match=PARENT_TRIGGER_ERROR,
    ):
        _insert_raw_binding(connection, _binding_values(case_id, claim.generation))


def test_binding_composite_foreign_key_rejects_predecessor_and_cascades_case_delete(
    settings: Settings,
) -> None:
    store, claim, result = _finished_case(settings, "case_binding_fk", bind=False)
    with connect_database(settings.workflow_db) as connection:
        trigger_definitions = _trigger_definitions(connection, "result_artifact_bindings")
        for name, _sql in trigger_definitions:
            connection.execute(f'DROP TRIGGER "{name}"')
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            _insert_raw_binding(
                connection,
                _binding_values(result.case_id, claim.generation + 1),
            )
        connection.rollback()
        for _name, sql in trigger_definitions:
            connection.execute(sql)
        connection.commit()
    store.save_result_artifact_binding(_binding(result.case_id, claim.generation), result)
    with connect_database(settings.workflow_db) as connection:
        connection.execute("DELETE FROM cases WHERE case_id = ?", (result.case_id,))
        connection.commit()
        assert connection.execute(
            "SELECT COUNT(*) FROM result_artifact_bindings WHERE case_id = ?",
            (result.case_id,),
        ).fetchone()[0] == 0
    with pytest.raises(InvoiceAgentsError) as excinfo:
        store.load_result_artifact_binding(result.case_id)
    assert excinfo.value.stop_reason == "CASE_NOT_FOUND"


@pytest.mark.parametrize(
    ("case_suffix", "assignments", "parameters"),
    [
        pytest.param(
            "running",
            "execution_state = 'RUNNING', lease_expires_at = ?",
            ((FINISHED_AT + timedelta(minutes=5)).isoformat(),),
            id="running",
        ),
        pytest.param("missing_result", "result_json = NULL", (), id="missing-result"),
        pytest.param("null_token", "execution_token = NULL", (), id="null-token"),
        pytest.param(
            "malformed_token",
            "execution_token = ?",
            ("exec_" + "g" * 32,),
            id="malformed-token",
        ),
        pytest.param(
            "finished_lease",
            "lease_expires_at = ?",
            ((FINISHED_AT + timedelta(minutes=5)).isoformat(),),
            id="finished-with-lease",
        ),
    ],
)
@pytest.mark.parametrize(
    "loader_name",
    ["load_result_artifact_binding", "load_result_with_artifact_binding"],
)
def test_load_binding_fails_loudly_for_corrupt_parent_state(
    loader_name: str,
    case_suffix: str,
    assignments: str,
    parameters: tuple[object, ...],
    settings: Settings,
) -> None:
    case_id = f"case_load_corrupt_{case_suffix}"
    store, _claim, _result = _finished_case(settings, case_id, bind=True)
    _corrupt_case(settings, case_id, assignments, parameters)

    loader = getattr(store, loader_name)
    with pytest.raises(InvoiceAgentsError) as excinfo:
        loader(case_id)
    assert excinfo.value.category == "DATABASE"
    assert excinfo.value.stop_reason == "PERSISTED_RESULT_INVALID"


def test_claiming_a_new_generation_invalidates_the_finished_binding(
    settings: Settings,
) -> None:
    store, first_claim, result = _finished_case(settings, "case_binding_reclaim", bind=True)

    second_claim = store.claim_case_execution(
        result.case_id,
        frozenset({CaseStatus.FAILED}),
        lease_seconds=60,
    )

    assert second_claim.generation == first_claim.generation + 1
    assert store.load_result_artifact_binding(result.case_id) is None


def test_replacing_a_finished_result_invalidates_its_binding(settings: Settings) -> None:
    store, claim, result = _finished_case(settings, "case_binding_refresh", bind=True)
    replacement = result.model_copy(
        update={
            "stop_reason": "BINDING_TEST_TWO",
            "finished_at": FINISHED_AT + timedelta(seconds=1),
        },
        deep=True,
    )

    store.update_finished_case_result(replacement, claim)

    assert store.load_result(result.case_id) == replacement
    assert store.load_result_artifact_binding(result.case_id) is None


@pytest.mark.parametrize("transition", ["handoff", "release", "finish", "recover"])
def test_every_running_transition_removes_a_defensively_detected_binding(
    transition: str,
    settings: Settings,
) -> None:
    case_id = f"case_binding_transition_{transition}"
    store, claim, source_id = _new_case(settings, case_id)
    recovered_at = FINISHED_AT + timedelta(minutes=10)
    if transition == "recover":
        with connect_database(settings.workflow_db) as connection:
            connection.execute(
                "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
                ((recovered_at - timedelta(seconds=1)).isoformat(), case_id),
            )
            connection.commit()
    _inject_binding_without_binding_triggers(settings, case_id, claim.generation)

    if transition == "handoff":
        store.handoff_case_execution(claim, lease_seconds=60)
    elif transition == "release":
        store.release_case_execution(claim)
    elif transition == "finish":
        store.finish_case(_terminal_result(case_id, source_id), claim)
    else:
        assert store.recover_expired_executions(now=recovered_at) == [case_id]

    assert store.load_result_artifact_binding(case_id) is None


@pytest.mark.parametrize("action", ["execute", "commit"])
@pytest.mark.parametrize("timing", ["before", "after"])
def test_binding_ordinary_kernel_fault_uses_exact_commit_readback(
    action: str,
    timing: str,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = f"case_binding_ordinary_{action}_{timing}"
    store, claim, result = _finished_case(settings, case_id, bind=False)
    binding = _binding(case_id, claim.generation)
    failure = OSError(f"binding {action} {timing} sentinel")
    _patch_binding_kernel_action(
        monkeypatch,
        action=action,
        timing=timing,
        failure=failure,
    )

    if action == "commit" and timing == "after":
        store.save_result_artifact_binding(binding, result)
        assert store.load_result_artifact_binding(case_id) == binding
    else:
        _assert_exact_failure(
            lambda: store.save_result_artifact_binding(binding, result),
            failure,
        )
        assert store.load_result_artifact_binding(case_id) is None


@pytest.mark.parametrize(
    "control_kind",
    ["keyboard-interrupt", "system-exit", "cancelled-error"],
)
@pytest.mark.parametrize("action", ["execute", "commit"])
@pytest.mark.parametrize("timing", ["before", "after"])
def test_binding_kernel_control_is_never_consumed_after_authoritative_readback(
    control_kind: str,
    action: str,
    timing: str,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = f"case_binding_control_{control_kind}_{action}_{timing}"
    store, claim, result = _finished_case(settings, case_id, bind=False)
    binding = _binding(case_id, claim.generation)
    control = _control_failure(control_kind, f"{action}-{timing}")
    _patch_binding_kernel_action(
        monkeypatch,
        action=action,
        timing=timing,
        failure=control,
    )

    _assert_exact_failure(
        lambda: store.save_result_artifact_binding(binding, result),
        control,
    )
    expected = binding if action == "commit" and timing == "after" else None
    assert store.load_result_artifact_binding(case_id) == expected


@pytest.mark.parametrize("rollback_timing", ["before", "after"])
def test_binding_rollback_fault_preserves_primary_when_readback_proves_absent(
    rollback_timing: str,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = f"case_binding_rollback_ordinary_{rollback_timing}"
    store, claim, result = _finished_case(settings, case_id, bind=False)
    binding = _binding(case_id, claim.generation)
    primary = OSError("binding insert primary sentinel")
    cleanup = OSError("binding rollback cleanup sentinel")
    _patch_binding_kernel_action(
        monkeypatch,
        action="execute",
        timing="after",
        failure=primary,
    )
    _patch_binding_kernel_action(
        monkeypatch,
        action="rollback",
        timing=rollback_timing,
        failure=cleanup,
    )

    _assert_exact_failure(
        lambda: store.save_result_artifact_binding(binding, result),
        primary,
    )
    assert store.load_result_artifact_binding(case_id) is None


@pytest.mark.parametrize(
    "cleanup_kind",
    ["keyboard-interrupt", "system-exit", "cancelled-error"],
)
@pytest.mark.parametrize("rollback_timing", ["before", "after"])
def test_binding_rollback_control_supersedes_ordinary_primary(
    cleanup_kind: str,
    rollback_timing: str,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = f"case_binding_rollback_control_{cleanup_kind}_{rollback_timing}"
    store, claim, result = _finished_case(settings, case_id, bind=False)
    binding = _binding(case_id, claim.generation)
    primary = OSError("binding insert ordinary primary")
    cleanup_control = _control_failure(cleanup_kind, "binding rollback control")
    _patch_binding_kernel_action(
        monkeypatch,
        action="execute",
        timing="after",
        failure=primary,
    )
    _patch_binding_kernel_action(
        monkeypatch,
        action="rollback",
        timing=rollback_timing,
        failure=cleanup_control,
    )

    _assert_exact_failure(
        lambda: store.save_result_artifact_binding(binding, result),
        cleanup_control,
    )
    assert store.load_result_artifact_binding(case_id) is None


@pytest.mark.parametrize(
    "primary_kind",
    ["keyboard-interrupt", "system-exit", "cancelled-error"],
)
@pytest.mark.parametrize("rollback_timing", ["before", "after"])
def test_binding_earliest_primary_control_survives_rollback_control(
    primary_kind: str,
    rollback_timing: str,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = f"case_binding_two_controls_{primary_kind}_{rollback_timing}"
    store, claim, result = _finished_case(settings, case_id, bind=False)
    binding = _binding(case_id, claim.generation)
    primary_control = _control_failure(primary_kind, "binding execute control")
    cleanup_control = SystemExit("binding rollback later control")
    _patch_binding_kernel_action(
        monkeypatch,
        action="execute",
        timing="after",
        failure=primary_control,
    )
    _patch_binding_kernel_action(
        monkeypatch,
        action="rollback",
        timing=rollback_timing,
        failure=cleanup_control,
    )

    _assert_exact_failure(
        lambda: store.save_result_artifact_binding(binding, result),
        primary_control,
    )
    assert store.load_result_artifact_binding(case_id) is None


@pytest.mark.parametrize(
    "primary_kind",
    ["keyboard-interrupt", "system-exit", "cancelled-error"],
)
@pytest.mark.parametrize("rollback_timing", ["before", "after"])
def test_binding_primary_control_survives_ordinary_rollback_failure(
    primary_kind: str,
    rollback_timing: str,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = f"case_binding_control_then_rollback_{primary_kind}_{rollback_timing}"
    store, claim, result = _finished_case(settings, case_id, bind=False)
    binding = _binding(case_id, claim.generation)
    primary_control = _control_failure(primary_kind, "binding execute control")
    rollback_failure = OSError("binding rollback ordinary failure")
    _patch_binding_kernel_action(
        monkeypatch,
        action="execute",
        timing="after",
        failure=primary_control,
    )
    _patch_binding_kernel_action(
        monkeypatch,
        action="rollback",
        timing=rollback_timing,
        failure=rollback_failure,
    )

    _assert_exact_failure(
        lambda: store.save_result_artifact_binding(binding, result),
        primary_control,
    )
    assert store.load_result_artifact_binding(case_id) is None


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("table", id="missing-table"),
        pytest.param("parent_index", id="missing-parent-index"),
        pytest.param("binding_insert_trigger", id="missing-binding-insert-trigger"),
        pytest.param("binding_update_trigger", id="missing-binding-update-trigger"),
        pytest.param("case_guard_trigger", id="missing-case-guard-trigger"),
    ],
)
def test_verification_rejects_each_binding_schema_object_mutation(
    mutation: str,
    settings: Settings,
) -> None:
    statements = {
        "table": "DROP TABLE result_artifact_bindings",
        "parent_index": "DROP INDEX idx_cases_case_generation",
        "binding_insert_trigger": (
            "DROP TRIGGER trg_result_artifact_bindings_terminal_insert"
        ),
        "binding_update_trigger": (
            "DROP TRIGGER trg_result_artifact_bindings_terminal_update"
        ),
        "case_guard_trigger": "DROP TRIGGER trg_cases_result_artifact_binding_guard_update",
    }
    with connect_database(settings.workflow_db) as connection:
        connection.execute(statements[mutation])
        connection.commit()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=settings,
        )
    assert excinfo.value.stop_reason == "DATABASE_SCHEMA_MISMATCH"


@pytest.mark.parametrize(
    "trigger_name",
    [
        "trg_cases_execution_token_grammar_insert",
        "trg_cases_execution_token_grammar_update",
    ],
)
@pytest.mark.parametrize("mutation", ["delete", "body"])
def test_latest_schema_verification_rejects_each_v4_token_grammar_trigger_mutation(
    trigger_name: str,
    mutation: str,
    settings: Settings,
) -> None:
    with connect_database(settings.workflow_db) as connection:
        row = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type = 'trigger' AND name = ?",
            (trigger_name,),
        ).fetchone()
        assert row is not None
        original_sql = str(row["sql"])
        connection.execute(f'DROP TRIGGER "{trigger_name}"')
        if mutation == "body":
            mutated_sql = original_sql.replace(
                "INVALID_EXECUTION_TOKEN",
                "INVALID_EXECUTION_TOKEN_MUTATED",
            )
            assert mutated_sql != original_sql
            connection.execute(mutated_sql)
        connection.commit()

    with pytest.raises(DatabaseVerificationError) as excinfo:
        verify_database(
            settings.workflow_db,
            DatabaseKind.WORKFLOW,
            settings=settings,
        )
    assert excinfo.value.stop_reason == "DATABASE_SCHEMA_MISMATCH"
