"""Durable, application-wide admission contracts for UI and CLI entrypoints.

The paid/model boundary is the only replaced dependency.  Source snapshots,
SQLite admission, execution claims, terminal writes, and recovery remain real.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, Lock, get_ident

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from typer.testing import CliRunner
from ui.factories import make_pending_review_case

from invoice_agents.cli import app as cli_app
from invoice_agents.config import Settings
from invoice_agents.db import store as store_module
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.hitl.service import record_human_decision
from invoice_agents.models import CaseResult, CaseStatus, ErrorRecord, HumanDecisionKind
from invoice_agents.ui import routes as ui_routes
from invoice_agents.ui import runs as ui_runs
from invoice_agents.ui import server as ui_server
from invoice_agents.ui.runs import BatchState, RunRegistry
from invoice_agents.ui.server import create_app


def _write_distinct_invoice(path: Path, invoice_number: str) -> Path:
    path.write_text(
        "\n".join(
            (
                "INVOICE",
                "",
                "Vendor: Admission Test Vendor",
                f"Invoice Number: {invoice_number}",
                "Date: 2026-08-10",
                "Due Date: 2026-08-25",
                "",
                "Items:",
                "  WidgetA    qty: 1    unit price: $10.00",
                "",
                "Subtotal: $10.00",
                "Tax (0%): $0.00",
                "Total Amount: $10.00",
                "",
                "Payment Terms: Net 15",
            )
        ),
        encoding="utf-8",
    )
    return path.resolve()


def _case_count(settings: Settings) -> int:
    with connect_database(settings.workflow_db, read_only=True) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0])


def _prepared_event_count(settings: Settings) -> int:
    with connect_database(settings.workflow_db, read_only=True) as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM events WHERE event_type = 'case.prepared'"
            ).fetchone()[0]
        )


def _event_count(settings: Settings) -> int:
    with connect_database(settings.workflow_db, read_only=True) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])


def _submission_row(settings: Settings, submission_id: str) -> sqlite3.Row:
    with connect_database(settings.workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT request_id, kind, fingerprint, redirect_target "
            "FROM submission_requests WHERE request_id = ?",
            (submission_id,),
        ).fetchone()
    assert row is not None
    return row


def _canonical_submission_fingerprint(kind: str, source_ids: list[str]) -> str:
    """Pin the durable contract to canonical JSON over kind and ordered source IDs."""

    canonical = json.dumps(
        {"kind": kind, "source_ids": source_ids},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _normalized_check_expressions(table_sql: str) -> tuple[str, ...]:
    """Extract exact normalized CHECK bodies without depending on SQL whitespace."""

    normalized = "".join(table_sql.lower().split())
    expressions: list[str] = []
    position = 0
    while (start := normalized.find("check(", position)) != -1:
        cursor = start + len("check(")
        expression_start = cursor
        depth = 1
        while cursor < len(normalized) and depth:
            if normalized[cursor] == "(":
                depth += 1
            elif normalized[cursor] == ")":
                depth -= 1
            cursor += 1
        assert depth == 0, table_sql
        expressions.append(normalized[expression_start : cursor - 1])
        position = cursor
    return tuple(expressions)


async def _await_event(event: asyncio.Event) -> None:
    async with asyncio.timeout(5):
        await event.wait()


async def _await_case(registry: RunRegistry, case_id: str) -> CaseResult:
    handle = registry.handle(case_id)
    assert handle is not None and handle.task is not None
    async with asyncio.timeout(10):
        return await asyncio.shield(handle.task)


async def _await_batch(batch: BatchState) -> None:
    assert batch.task is not None
    async with asyncio.timeout(10):
        await asyncio.shield(batch.task)


def _finish_success(
    case_id: str,
    started_at: datetime,
    settings: Settings,
    claim: ExecutionClaim | None,
) -> CaseResult:
    assert claim is not None
    store = WorkflowStore(settings)
    result = CaseResult(
        case_id=case_id,
        source_id=store.load_authoritative_case_source_id(claim),
        status=CaseStatus.SUCCEEDED,
        stop_reason="ADMISSION_TEST_MODEL_FINISHED",
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )
    store.finish_case(result, claim)
    return result


def _successful_model(
    calls: list[str],
    *,
    entered: asyncio.Event | None = None,
    release: asyncio.Event | None = None,
) -> Callable[..., Awaitable[CaseResult]]:
    async def run(
        case_id: str,
        started_at: datetime,
        settings: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        calls.append(case_id)
        if entered is not None:
            entered.set()
        if release is not None:
            await release.wait()
        return _finish_success(case_id, started_at, settings, claim)

    return run


def test_workflow_v6_installs_durable_admission_schema(settings: Settings) -> None:
    """Renumbering Task 11 as v5 would overwrite Task 9's immutable migration."""

    with connect_database(settings.workflow_db, read_only=True) as connection:
        versions = tuple(
            int(row[0])
            for row in connection.execute("SELECT version FROM schema_version ORDER BY version")
        )
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            ).fetchall()
        }
        columns = {
            table: tuple(
                (
                    str(row[1]),
                    str(row[2]),
                    int(row[3]),
                    None if row[4] is None else str(row[4]),
                    int(row[5]),
                )
                for row in connection.execute(f"PRAGMA table_info({table})")
            )
            for table in (
                "submission_requests",
                "source_run_claims",
                "batches",
                "batch_entries",
            )
        }
        foreign_keys = {
            table: {
                (
                    str(row[3]),
                    str(row[2]),
                    str(row[4]),
                    str(row[5]),
                    str(row[6]),
                    str(row[7]),
                )
                for row in connection.execute(f"PRAGMA foreign_key_list({table})")
            }
            for table in columns
        }
        schema_sql = {
            table: "".join(str(row[0]).lower().split())
            for table in columns
            if (
                row := connection.execute(
                    "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = ?",
                    (table,),
                ).fetchone()
            )
            is not None
        }
        indexes: dict[str, set[tuple[str, int, int, tuple[str, ...]]]] = {}
        for table in columns:
            indexes[table] = set()
            for index in connection.execute(f"PRAGMA index_list({table})"):
                name = str(index[1])
                origin = str(index[3])
                stable_name = name if origin == "c" else f"<{origin}>"
                indexes[table].add(
                    (
                        stable_name,
                        int(index[2]),
                        int(index[4]),
                        tuple(
                            str(column[2])
                            for column in connection.execute(f"PRAGMA index_info('{name}')")
                        ),
                    )
                )

    assert versions == (1, 2, 3, 4, 5, 6)
    assert {
        "submission_requests",
        "source_run_claims",
        "batches",
        "batch_entries",
    } <= tables
    assert columns == {
        "submission_requests": (
            ("request_id", "TEXT", 0, None, 1),
            ("created_at", "TEXT", 1, None, 0),
            ("kind", "TEXT", 1, None, 0),
            ("fingerprint", "TEXT", 1, None, 0),
            ("redirect_target", "TEXT", 1, None, 0),
        ),
        "source_run_claims": (
            ("source_id", "TEXT", 0, None, 1),
            ("case_id", "TEXT", 1, None, 0),
            ("state", "TEXT", 1, None, 0),
            ("claimed_at", "TEXT", 1, None, 0),
            ("released_at", "TEXT", 0, None, 0),
        ),
        "batches": (
            ("batch_id", "TEXT", 0, None, 1),
            ("created_at", "TEXT", 1, None, 0),
            ("concurrency", "INTEGER", 1, None, 0),
            ("state", "TEXT", 1, None, 0),
        ),
        "batch_entries": (
            ("batch_id", "TEXT", 1, None, 1),
            ("position", "INTEGER", 1, None, 2),
            ("source_id", "TEXT", 1, None, 0),
            ("case_id", "TEXT", 1, None, 0),
            ("source_path", "TEXT", 1, None, 0),
            ("state", "TEXT", 1, None, 0),
        ),
    }
    assert foreign_keys == {
        "submission_requests": set(),
        "source_run_claims": {
            ("source_id", "source_artifacts", "source_id", "NO ACTION", "NO ACTION", "NONE"),
            ("case_id", "cases", "case_id", "NO ACTION", "NO ACTION", "NONE"),
        },
        "batches": set(),
        "batch_entries": {
            ("batch_id", "batches", "batch_id", "NO ACTION", "NO ACTION", "NONE"),
            ("source_id", "source_artifacts", "source_id", "NO ACTION", "NO ACTION", "NONE"),
            ("case_id", "cases", "case_id", "NO ACTION", "NO ACTION", "NONE"),
        },
    }
    assert {table: _normalized_check_expressions(sql) for table, sql in schema_sql.items()} == {
        "submission_requests": (
            "kindin('single','batch')",
            "length(fingerprint)=64andfingerprintnotglob'*[^0-9a-f]*'",
        ),
        "source_run_claims": ("statein('queued','running','done','failed')",),
        "batches": (
            "concurrencybetween1and8",
            "statein('queued','running','done','failed')",
        ),
        "batch_entries": (
            "position>=0",
            "statein('queued','running','done','failed')",
        ),
    }
    assert indexes == {
        "submission_requests": {("<pk>", 1, 0, ("request_id",))},
        "source_run_claims": {
            ("<pk>", 1, 0, ("source_id",)),
            ("idx_source_run_claims_case_id", 0, 0, ("case_id",)),
        },
        "batches": {
            ("<pk>", 1, 0, ("batch_id",)),
            ("idx_batches_state", 0, 0, ("state",)),
        },
        "batch_entries": {
            ("<pk>", 1, 0, ("batch_id", "position")),
            ("<u>", 1, 0, ("batch_id", "source_id")),
            ("idx_batch_entries_case_id", 0, 0, ("case_id",)),
        },
    }


def test_run_registry_requires_an_explicit_application_limit() -> None:
    """Restoring a constructor default would silently bypass application policy."""

    with pytest.raises(TypeError):
        RunRegistry()


def test_force_reprocess_flag_reaches_the_single_source_admission_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping the CLI-only force flag would make deliberate reprocessing impossible."""

    source = _write_distinct_invoice(tmp_path / "cli-force.txt", "CLI-FORCE-1")
    observed: list[tuple[Path, bool]] = []

    async def process(
        path: Path,
        _settings: Settings,
        *,
        force_reprocess: bool = False,
    ) -> CaseResult:
        observed.append((path.resolve(), force_reprocess))
        now = datetime.now(UTC)
        return CaseResult(
            case_id="case_cli_force_contract",
            source_id="src_cli_force_contract",
            status=CaseStatus.SUCCEEDED,
            stop_reason="CLI_FORCE_CONTRACT_FINISHED",
            started_at=now,
            finished_at=now,
        )

    monkeypatch.setattr("invoice_agents.cli.process_invoice", process)
    result = CliRunner().invoke(
        cli_app,
        ["process", "--invoice-path", str(source), "--force-reprocess"],
    )

    assert result.exit_code == 0
    assert observed == [(source, True)]


def test_cli_batch_reuses_durable_terminal_source_claims(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI batch must not bypass durable admission through legacy process_batch."""

    invoice_dir = tmp_path / "cli-batch"
    invoice_dir.mkdir()
    paths = [
        _write_distinct_invoice(invoice_dir / f"invoice-{index}.txt", f"CLI-BATCH-{index}")
        for index in range(2)
    ]
    calls: list[str] = []
    monkeypatch.setattr("invoice_agents.cli._settings", lambda: settings)
    monkeypatch.setattr(ui_runs, "run_prepared_case", _successful_model(calls))

    first = CliRunner().invoke(
        cli_app,
        ["batch", "--invoice-dir", str(invoice_dir), "--concurrency", "1"],
    )
    assert first.exit_code == 0, first.output
    first_cases = tuple(calls)
    assert len(first_cases) == len(paths)

    repeated = CliRunner().invoke(
        cli_app,
        ["batch", "--invoice-dir", str(invoice_dir), "--concurrency", "2"],
    )
    assert repeated.exit_code == 0, repeated.output
    assert tuple(calls) == first_cases
    assert _case_count(settings) == len(paths)


@pytest.mark.asyncio
async def test_create_app_registry_budget_spans_single_batch_and_resume_routes(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-request or per-route registries would exceed the configured global limit."""

    invoice_dir = tmp_path / "data" / "invoices"
    invoice_dir.mkdir(parents=True)
    paths = [
        _write_distinct_invoice(invoice_dir / f"global-{index}.txt", f"GLOBAL-{index}")
        for index in range(3)
    ]
    resume_case_id, review = make_pending_review_case(settings)
    record_human_decision(
        review.review_id,
        "admission reviewer",
        HumanDecisionKind.REJECT,
        "the deterministic admission test rejects this invoice",
        WorkflowStore(settings),
        settings.inventory_db,
    )
    monkeypatch.chdir(tmp_path)
    active = 0
    peak = 0
    calls: list[str] = []
    boundary_tasks: dict[str, asyncio.Task[object]] = {}
    constructions: list[tuple[int, int, int]] = []
    semaphore_instances: list[asyncio.Semaphore] = []
    acquisitions: list[tuple[asyncio.Semaphore, asyncio.Task[object]]] = []
    at_capacity = asyncio.Event()
    release = asyncio.Event()

    real_semaphore_init = asyncio.Semaphore.__init__
    real_semaphore_acquire = asyncio.Semaphore.acquire

    def observed_semaphore_init(self: asyncio.Semaphore, value: int = 1) -> None:
        real_semaphore_init(self, value)
        semaphore_instances.append(self)

    async def observed_semaphore_acquire(self: asyncio.Semaphore) -> bool:
        task = asyncio.current_task()
        assert task is not None
        acquisitions.append((self, task))
        return await real_semaphore_acquire(self)

    real_registry_init = RunRegistry.__init__

    def observed_registry_init(self: RunRegistry, *, global_limit: int) -> None:
        real_registry_init(self, global_limit=global_limit)
        slots = self._model_slots
        constructions.append((global_limit, id(slots), slots._value))

    monkeypatch.setattr(RunRegistry, "__init__", observed_registry_init)

    async def enter_boundary(label: str) -> None:
        nonlocal active, peak
        task = asyncio.current_task()
        assert task is not None
        boundary_tasks[label] = task
        active += 1
        peak = max(peak, active)
        calls.append(label)
        if active == 2:
            at_capacity.set()
        await release.wait()
        active -= 1

    async def run_case(
        case_id: str,
        started_at: datetime,
        selected: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        await enter_boundary(f"run:{case_id}")
        return _finish_success(case_id, started_at, selected, claim)

    async def resume(
        case_id: str,
        selected: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        previous = WorkflowStore(selected).load_result(case_id)
        assert previous is not None
        await enter_boundary(f"resume:{case_id}")
        return _finish_success(case_id, previous.started_at, selected, claim)

    monkeypatch.setattr(ui_runs, "run_prepared_case", run_case)
    monkeypatch.setattr(ui_runs, "resume_case", resume)
    monkeypatch.setattr(asyncio.Semaphore, "__init__", observed_semaphore_init)
    monkeypatch.setattr(asyncio.Semaphore, "acquire", observed_semaphore_acquire)
    settings.case_concurrency = 2
    app = ui_server.create_app(settings)
    registry = app.state.registry
    assert constructions == [(settings.case_concurrency, id(registry._model_slots), 2)]

    def request(path: str) -> Request:
        return Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode(),
                "query_string": b"",
                "headers": (),
                "client": ("127.0.0.1", 1),
                "server": ("testserver", 80),
                "app": app,
            }
        )

    single_response = await ui_routes.submit_invoice(
        request("/submit"),
        existing=[paths[0].name],
        upload=None,
        submission_id="submission_global_single",
    )
    assert single_response.status_code == 303
    single = single_response.headers["location"].split("/")[2]
    paths[0].unlink()
    batch_response = await ui_routes.submit_batch(
        request("/batch"),
        concurrency=8,
        submission_id="submission_global_batch",
    )
    assert batch_response.status_code == 303
    batch_id = batch_response.headers["location"].split("/")[2]
    batch = registry.batch(batch_id)
    assert batch is not None
    resume_response = await ui_routes.case_resume(
        request(f"/cases/{resume_case_id}/resume"), resume_case_id
    )
    assert resume_response.status_code == 303
    resume_handle = registry.handle(resume_case_id)
    assert resume_handle is not None

    await _await_event(at_capacity)
    turn_completed = asyncio.Event()
    asyncio.get_running_loop().call_soon(turn_completed.set)
    await _await_event(turn_completed)
    assert active == 2
    assert len(calls) == 2

    release.set()
    await _await_case(registry, single)
    await _await_batch(batch)
    assert resume_handle.task is not None
    async with asyncio.timeout(10):
        await asyncio.shield(resume_handle.task)

    assert peak == 2
    assert len(calls) == 4
    assert sum(label.startswith("resume:") for label in calls) == 1
    assert constructions == [(settings.case_concurrency, id(registry._model_slots), 2)]
    assert semaphore_instances == [registry._model_slots]
    assert len(acquisitions) == len(boundary_tasks) == 4
    for label, boundary_task in boundary_tasks.items():
        assert [
            semaphore
            for semaphore, acquiring_task in acquisitions
            if acquiring_task is boundary_task
        ] == [registry._model_slots], label
    assert app.state.registry is registry


@pytest.mark.asyncio
async def test_batch_requested_concurrency_is_enforced_below_the_application_limit(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisting a batch limit without enforcing it would misrepresent paid-work admission."""

    paths = [
        _write_distinct_invoice(tmp_path / f"batch-limit-{index}.txt", f"BATCH-LIMIT-{index}")
        for index in range(3)
    ]
    active = 0
    peak = 0
    calls: list[str] = []
    first_entered = asyncio.Event()
    release = asyncio.Event()

    async def model(
        case_id: str,
        started_at: datetime,
        selected: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append(case_id)
        first_entered.set()
        try:
            await release.wait()
            return _finish_success(case_id, started_at, selected, claim)
        finally:
            active -= 1

    monkeypatch.setattr(ui_runs, "run_prepared_case", model)
    batch = await RunRegistry(global_limit=3).start_batch(
        paths,
        settings,
        concurrency=1,
        submission_id="submission_batch_limit_one",
    )
    await _await_event(first_entered)
    turn_completed = asyncio.Event()
    asyncio.get_running_loop().call_soon(turn_completed.set)
    await _await_event(turn_completed)
    assert active == 1
    assert len(calls) == 1

    release.set()
    await _await_batch(batch)
    assert peak == 1
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_bounded_batch_failure_terminalizes_claims_that_never_reached_a_worker(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded scheduler must retain durability ownership for unscheduled entries."""

    paths = [
        _write_distinct_invoice(tmp_path / f"unscheduled-{index}.txt", f"UNSCHEDULED-{index}")
        for index in range(3)
    ]
    calls: list[str] = []

    async def fail_first_model(
        case_id: str,
        _started_at: datetime,
        _selected: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        assert claim is not None
        calls.append(case_id)
        raise RuntimeError("test bounded worker failure")

    monkeypatch.setattr(ui_runs, "run_prepared_case", fail_first_model)
    registry = RunRegistry(global_limit=3)
    batch = await registry.start_batch(
        paths,
        settings,
        concurrency=1,
        submission_id="submission_unscheduled_failure",
    )
    task = batch.task
    assert task is not None
    with pytest.raises(RuntimeError, match="test bounded worker failure"):
        async with asyncio.timeout(10):
            await asyncio.shield(task)

    with connect_database(settings.workflow_db, read_only=True) as connection:
        case_rows = connection.execute(
            "SELECT case_id, execution_state FROM cases ORDER BY case_id"
        ).fetchall()
        claim_rows = connection.execute(
            "SELECT state FROM source_run_claims ORDER BY source_id"
        ).fetchall()
        entry_rows = connection.execute(
            "SELECT state FROM batch_entries ORDER BY position"
        ).fetchall()
        batch_row = connection.execute(
            "SELECT state FROM batches WHERE batch_id = ?", (batch.batch_id,)
        ).fetchone()
    assert len(calls) == 1
    assert [row["execution_state"] for row in case_rows] == ["FINISHED"] * 3
    assert [row["state"] for row in claim_rows] == ["failed"] * 3
    assert [row["state"] for row in entry_rows] == ["failed"] * 3
    assert batch_row is not None and batch_row["state"] == "failed"
    assert registry._lifecycle_owners == {}


@pytest.mark.asyncio
async def test_duplicate_submission_reuses_one_case_and_durable_exact_target(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two real BEGIN IMMEDIATE contenders must converge on normalized source identity."""

    source = _write_distinct_invoice(tmp_path / "duplicate.txt", "DUPLICATE-1")
    alias = tmp_path / "same-bytes-different-path.txt"
    alias.write_bytes(source.read_bytes())
    alias = alias.resolve()
    calls: list[str] = []
    calls_lock = Lock()
    race = Barrier(3, timeout=5)
    seen_threads: set[int] = set()
    seen_lock = Lock()
    real_connect = store_module.connect_database

    @contextmanager
    def observed_connect(*args: object, **kwargs: object):
        with real_connect(*args, **kwargs) as connection:
            if not kwargs.get("read_only", False):

                def trace(statement: str) -> None:
                    if statement.strip().upper() != "BEGIN IMMEDIATE":
                        return
                    identity = get_ident()
                    with seen_lock:
                        if identity in seen_threads:
                            return
                        seen_threads.add(identity)
                    race.wait()

                connection.set_trace_callback(trace)
            yield connection

    async def model(
        case_id: str,
        started_at: datetime,
        selected: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        with calls_lock:
            calls.append(case_id)
        return _finish_success(case_id, started_at, selected, claim)

    async def admit(path: Path, registry: RunRegistry) -> str:
        outcome = await registry.start_process(
            path,
            settings,
            submission_id="submission_duplicate_single",
        )
        assert isinstance(outcome, str)
        handle = registry.handle(outcome)
        if handle is not None and handle.task is not None:
            await _await_case(registry, outcome)
        return outcome

    monkeypatch.setattr(store_module, "connect_database", observed_connect)
    monkeypatch.setattr(ui_runs, "run_prepared_case", model)
    registries = [RunRegistry(global_limit=2), RunRegistry(global_limit=2)]
    contenders = [
        asyncio.create_task(asyncio.to_thread(asyncio.run, admit(path, registry)))
        for path, registry in zip((source, alias), registries, strict=True)
    ]
    await asyncio.to_thread(race.wait)
    first, duplicate = await asyncio.gather(*contenders)

    assert duplicate == first
    assert calls == [first]
    assert _case_count(settings) == 1
    assert _prepared_event_count(settings) == 1
    row = _submission_row(settings, "submission_duplicate_single")
    assert row["kind"] == "single"
    assert row["redirect_target"] == f"/cases/{first}/live"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        persisted_case = connection.execute(
            "SELECT source_id FROM cases WHERE case_id = ?", (first,)
        ).fetchone()
    assert persisted_case is not None
    assert row["fingerprint"] == _canonical_submission_fingerprint(
        "single", [str(persisted_case["source_id"])]
    )

    monkeypatch.setattr(store_module, "connect_database", real_connect)
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE submission_requests SET redirect_target = ? WHERE request_id = ?",
            ("/cases/case_forged/live", "submission_duplicate_single"),
        )
        connection.commit()
    with pytest.raises(InvoiceAgentsError) as corrupt_target:
        await RunRegistry(global_limit=2).start_process(
            alias,
            settings,
            submission_id="submission_duplicate_single",
        )
    assert corrupt_target.value.stop_reason == "PERSISTED_SUBMISSION_INVALID"
    assert calls == [first]
    assert _case_count(settings) == 1
    assert _prepared_event_count(settings) == 1


@pytest.mark.parametrize(
    "corruption",
    (
        "missing",
        "state",
        "claimed_at",
        "released_at",
        "released_at_grammar",
        "released_at_order",
        "case_binding",
    ),
)
@pytest.mark.asyncio
async def test_duplicate_single_submission_rejects_corrupt_source_claim(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    """An idempotent redirect must not conceal corrupt source-run authority."""

    first = _write_distinct_invoice(tmp_path / "claim-first.txt", "CLAIM-FIRST")
    second = _write_distinct_invoice(tmp_path / "claim-second.txt", "CLAIM-SECOND")
    calls: list[str] = []
    monkeypatch.setattr(ui_runs, "run_prepared_case", _successful_model(calls))
    registry = RunRegistry(global_limit=2)
    first_case = await registry.start_process(
        first,
        settings,
        submission_id="submission_claim_corruption",
    )
    assert isinstance(first_case, str)
    await _await_case(registry, first_case)
    second_case = await registry.start_process(
        second,
        settings,
        submission_id="submission_claim_corruption_fixture",
    )
    assert isinstance(second_case, str)
    await _await_case(registry, second_case)

    with connect_database(settings.workflow_db) as connection:
        source_id = str(
            connection.execute(
                "SELECT source_id FROM cases WHERE case_id = ?", (first_case,)
            ).fetchone()[0]
        )
        if corruption == "missing":
            connection.execute("DELETE FROM source_run_claims WHERE source_id = ?", (source_id,))
        elif corruption == "state":
            connection.execute(
                "UPDATE source_run_claims SET state = 'running', released_at = NULL "
                "WHERE source_id = ?",
                (source_id,),
            )
        elif corruption == "claimed_at":
            connection.execute(
                "UPDATE source_run_claims SET claimed_at = 'not-a-time' WHERE source_id = ?",
                (source_id,),
            )
        elif corruption == "released_at":
            connection.execute(
                "UPDATE source_run_claims SET released_at = NULL WHERE source_id = ?",
                (source_id,),
            )
        elif corruption == "released_at_grammar":
            connection.execute(
                "UPDATE source_run_claims SET released_at = 'not-a-time' WHERE source_id = ?",
                (source_id,),
            )
        elif corruption == "released_at_order":
            connection.execute(
                "UPDATE source_run_claims SET released_at = claimed_at WHERE source_id = ?",
                (source_id,),
            )
        else:
            connection.execute(
                "UPDATE source_run_claims SET case_id = ? WHERE source_id = ?",
                (second_case, source_id),
            )
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as corrupt_claim:
        await RunRegistry(global_limit=2).start_process(
            first,
            settings,
            submission_id="submission_claim_corruption",
        )
    assert corrupt_claim.value.stop_reason == "PERSISTED_SUBMISSION_INVALID"
    assert calls == [first_case, second_case]
    assert _case_count(settings) == 2


@pytest.mark.asyncio
async def test_submission_id_rejects_kind_and_order_changes_without_mutation(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparing only an unordered source set would admit a different request."""

    first_path = _write_distinct_invoice(tmp_path / "ordered-a.txt", "ORDER-A")
    second_path = _write_distinct_invoice(tmp_path / "ordered-b.txt", "ORDER-B")
    calls: list[str] = []
    monkeypatch.setattr(ui_runs, "run_prepared_case", _successful_model(calls))
    registry = RunRegistry(global_limit=2)
    batch = await registry.start_batch(
        [first_path, second_path],
        settings,
        concurrency=2,
        submission_id="submission_ordered_batch",
    )
    await _await_batch(batch)
    before = tuple(_submission_row(settings, "submission_ordered_batch"))
    with connect_database(settings.workflow_db, read_only=True) as connection:
        ordered_source_ids = [
            str(row["source_id"])
            for row in connection.execute(
                "SELECT source_id FROM batch_entries WHERE batch_id = ? ORDER BY position",
                (batch.batch_id,),
            ).fetchall()
        ]
    assert before[1] == "batch"
    assert before[2] == _canonical_submission_fingerprint("batch", ordered_source_ids)
    assert before[3] == f"/batches/{batch.batch_id}"

    with pytest.raises(InvoiceAgentsError) as reversed_error:
        await RunRegistry(global_limit=2).start_batch(
            [second_path, first_path],
            settings,
            concurrency=2,
            submission_id="submission_ordered_batch",
        )
    assert reversed_error.value.stop_reason == "SUBMISSION_FINGERPRINT_MISMATCH"

    with pytest.raises(InvoiceAgentsError) as kind_error:
        await RunRegistry(global_limit=2).start_process(
            first_path,
            settings,
            submission_id="submission_ordered_batch",
        )
    assert kind_error.value.stop_reason == "SUBMISSION_FINGERPRINT_MISMATCH"
    assert tuple(_submission_row(settings, "submission_ordered_batch")) == before
    assert _case_count(settings) == 2
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("fault_table", "source_count"),
    (("submission_requests", 1), ("batches", 2)),
)
@pytest.mark.asyncio
async def test_source_case_claim_and_target_admission_roll_back_together(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_table: str,
    source_count: int,
) -> None:
    """Committing before the durable target row would leave paid-work authority behind."""

    sources = [
        _write_distinct_invoice(tmp_path / f"atomic-{index}.txt", f"ATOMIC-{index}")
        for index in range(source_count)
    ]
    prerequisite = (
        f"(SELECT COUNT(*) FROM source_artifacts) = {source_count} "
        f"AND (SELECT COUNT(*) FROM cases WHERE execution_state = 'RUNNING') = {source_count} "
        f"AND (SELECT COUNT(*) FROM source_run_claims) = {source_count}"
    )
    if fault_table == "batches":
        prerequisite += " AND (SELECT COUNT(*) FROM submission_requests) = 1"
    trigger_calls: list[str] = []
    real_connect = store_module.connect_database

    def fail_exact_late_insert() -> int:
        trigger_calls.append(fault_table)
        raise sqlite3.IntegrityError(f"test late {fault_table} admission fault")

    @contextmanager
    def fault_observing_connect(path: Path, *, read_only: bool = False):
        with real_connect(path, read_only=read_only) as selected_connection:
            if not read_only:
                selected_connection.create_function(
                    "test_late_admission_fault",
                    0,
                    fail_exact_late_insert,
                )
            yield selected_connection

    with connect_database(settings.workflow_db) as connection:
        connection.executescript(
            f"""
            CREATE TRIGGER test_abort_late_admission
            BEFORE INSERT ON {fault_table}
            WHEN {prerequisite}
            BEGIN
                SELECT test_late_admission_fault();
            END;
            """
        )
        connection.commit()

    async def forbidden_model(*_args: object, **_kwargs: object) -> CaseResult:
        raise AssertionError("failed admission reached the model boundary")

    monkeypatch.setattr(store_module, "connect_database", fault_observing_connect)
    monkeypatch.setattr(ui_runs, "run_prepared_case", forbidden_model)
    with pytest.raises((InvoiceAgentsError, sqlite3.DatabaseError)):
        registry = RunRegistry(global_limit=2)
        if fault_table == "submission_requests":
            await registry.start_process(
                sources[0],
                settings,
                submission_id="submission_atomic_single_failure",
            )
        else:
            await registry.start_batch(
                sources,
                settings,
                concurrency=2,
                submission_id="submission_atomic_batch_failure",
            )
    assert trigger_calls == [fault_table]

    with connect_database(settings.workflow_db, read_only=True) as connection:
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "source_artifacts",
                "cases",
                "submission_requests",
                "source_run_claims",
                "batches",
                "batch_entries",
            )
        }
    assert counts == {
        "source_artifacts": 0,
        "cases": 0,
        "submission_requests": 0,
        "source_run_claims": 0,
        "batches": 0,
        "batch_entries": 0,
    }


@pytest.mark.asyncio
async def test_batch_handoff_failure_terminalizes_every_committed_fresh_claim(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-commit owner-install failure must not orphan later batch claims."""

    sources = [
        _write_distinct_invoice(tmp_path / f"handoff-{index}.txt", f"HANDOFF-{index}")
        for index in range(3)
    ]
    calls: list[str] = []
    registry = RunRegistry(global_limit=3)
    real_install = registry._install_lifecycle_owner
    installs = 0

    def fail_second_install(
        case_id: str,
        started_at: datetime,
        claim: ExecutionClaim,
    ) -> object:
        nonlocal installs
        installs += 1
        if installs == 2:
            raise RuntimeError("test batch lifecycle handoff failure")
        return real_install(case_id, started_at, claim)

    monkeypatch.setattr(ui_runs, "run_prepared_case", _successful_model(calls))
    monkeypatch.setattr(registry, "_install_lifecycle_owner", fail_second_install)
    with pytest.raises(RuntimeError, match="test batch lifecycle handoff failure"):
        await registry.start_batch(
            sources,
            settings,
            concurrency=3,
            submission_id="submission_handoff_failure",
        )

    with connect_database(settings.workflow_db, read_only=True) as connection:
        case_rows = connection.execute(
            "SELECT execution_state FROM cases ORDER BY case_id"
        ).fetchall()
        claim_rows = connection.execute(
            "SELECT state, released_at FROM source_run_claims ORDER BY source_id"
        ).fetchall()
        entry_rows = connection.execute(
            "SELECT state FROM batch_entries ORDER BY position"
        ).fetchall()
        batch_row = connection.execute("SELECT state FROM batches").fetchone()
    assert [row["execution_state"] for row in case_rows] == ["FINISHED"] * 3
    assert [row["state"] for row in claim_rows] == ["failed"] * 3
    assert all(row["released_at"] is not None for row in claim_rows)
    assert [row["state"] for row in entry_rows] == ["failed"] * 3
    assert batch_row is not None and batch_row["state"] == "failed"
    assert registry._lifecycle_owners == {}
    assert calls == []


@pytest.mark.parametrize(
    "errors",
    (
        [],
        [ErrorRecord(category="NOT_A_REAL_CATEGORY", message="invalid staging category")],
    ),
    ids=("missing-error-evidence", "unknown-error-category"),
)
@pytest.mark.asyncio
async def test_batch_rejects_malformed_staging_failure_evidence(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    errors: list[ErrorRecord],
) -> None:
    """Malformed failure evidence must not be rewritten as an ordinary failure."""

    source = _write_distinct_invoice(tmp_path / "malformed-stage.txt", "MALFORMED-STAGE")
    failed_at = datetime.now(UTC)

    async def malformed_stage(_path: Path, _settings: Settings) -> CaseResult:
        return CaseResult(
            case_id="case_00000000000000000000000000000000",
            source_id=None,
            status=CaseStatus.FAILED,
            stop_reason="MALFORMED_STAGING_FAILURE",
            errors=errors,
            started_at=failed_at,
            finished_at=failed_at,
        )

    monkeypatch.setattr(ui_runs, "_prepare_claimed_for_launch", malformed_stage)
    with pytest.raises(InvoiceAgentsError) as invalid:
        await RunRegistry(global_limit=1).start_batch(
            [source],
            settings,
            concurrency=1,
            submission_id="submission_malformed_staging_failure",
        )

    assert invalid.value.category is ErrorCategory.ORCHESTRATION
    assert invalid.value.stop_reason == "PREPARATION_WORKER_PROTOCOL_INVALID"
    assert _case_count(settings) == 0


@pytest.mark.asyncio
async def test_batch_handoff_rejects_a_mismatched_preexisting_private_owner(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-commit repair must identify, not silently retain, wrong private authority."""

    source = _write_distinct_invoice(tmp_path / "owner-mismatch.txt", "OWNER-MISMATCH")
    registry = RunRegistry(global_limit=1)
    real_install = registry._install_lifecycle_owner

    def install_mismatched_owner(
        case_id: str,
        started_at: datetime,
        claim: ExecutionClaim,
    ) -> object:
        wrong_claim = ExecutionClaim(
            case_id=case_id,
            token="exec_00000000000000000000000000000000",
            generation=claim.generation,
            expires_at=claim.expires_at,
        )
        assert wrong_claim != claim
        registry._lifecycle_owners[case_id] = ui_runs._LifecycleOwner(
            claim=wrong_claim,
            started_at=started_at,
        )
        return real_install(case_id, started_at, claim)

    monkeypatch.setattr(registry, "_install_lifecycle_owner", install_mismatched_owner)
    with pytest.raises(InvoiceAgentsError) as mismatch:
        await registry.start_batch(
            [source],
            settings,
            concurrency=1,
            submission_id="submission_owner_mismatch",
        )
    assert mismatch.value.stop_reason == "CASE_LIFECYCLE_OWNER_MISMATCH"

    with connect_database(settings.workflow_db, read_only=True) as connection:
        case_row = connection.execute("SELECT execution_state FROM cases").fetchone()
        claim_row = connection.execute(
            "SELECT state, released_at FROM source_run_claims"
        ).fetchone()
    assert case_row is not None and case_row["execution_state"] == "FINISHED"
    assert claim_row is not None and claim_row["state"] == "failed"
    assert claim_row["released_at"] is not None
    retained = next(iter(registry._lifecycle_owners.values()))
    assert retained.claim.token == "exec_00000000000000000000000000000000"


@pytest.mark.asyncio
async def test_batch_cancellation_before_runner_ownership_mirrors_every_terminal_admission(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling before the batch runner starts must not leave active admission rows."""

    sources = [
        _write_distinct_invoice(tmp_path / f"early-cancel-{index}.txt", f"EARLY-{index}")
        for index in range(2)
    ]
    registry = RunRegistry(global_limit=2)
    runner_entered = asyncio.Event()
    hold_runner = asyncio.Event()

    async def paused_runner(
        _batch: BatchState,
        _entries: list[ui_runs.BatchEntry],
        _settings: Settings,
        _ownership_installed: asyncio.Event,
    ) -> None:
        runner_entered.set()
        await hold_runner.wait()

    monkeypatch.setattr(registry, "_run_batch", paused_runner)
    start = asyncio.create_task(
        registry.start_batch(
            sources,
            settings,
            concurrency=2,
            submission_id="submission_cancel_before_runner_ownership",
        )
    )
    await _await_event(runner_entered)
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start

    with connect_database(settings.workflow_db, read_only=True) as connection:
        cases = connection.execute("SELECT execution_state FROM cases ORDER BY case_id").fetchall()
        claims = connection.execute(
            "SELECT state, released_at FROM source_run_claims ORDER BY source_id"
        ).fetchall()
        entries = connection.execute("SELECT state FROM batch_entries ORDER BY position").fetchall()
        batch = connection.execute("SELECT state FROM batches").fetchone()
    assert [row["execution_state"] for row in cases] == ["FINISHED", "FINISHED"]
    assert [row["state"] for row in claims] == ["failed", "failed"]
    assert all(row["released_at"] is not None for row in claims)
    assert [row["state"] for row in entries] == ["failed", "failed"]
    assert batch is not None and batch["state"] == "failed"
    assert registry._lifecycle_owners == {}


@pytest.mark.asyncio
async def test_admission_mirror_failure_precedes_model_failure_and_retains_owner(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution durability alone must not conceal a failed admission-state mirror."""

    source = _write_distinct_invoice(tmp_path / "mirror-failure.txt", "MIRROR-FAILURE")
    admission_failure = InvoiceAgentsError(
        ErrorCategory.DATABASE,
        "test admission mirror failure",
        stop_reason="PERSISTED_SUBMISSION_INVALID",
    )

    async def fail_model(*_args: object, **_kwargs: object) -> CaseResult:
        raise RuntimeError("test model failure before terminal persistence")

    def fail_admission_mirror(_self: WorkflowStore, _case_id: str) -> None:
        raise admission_failure

    monkeypatch.setattr(ui_runs, "run_prepared_case", fail_model)
    monkeypatch.setattr(WorkflowStore, "mark_admission_terminal", fail_admission_mirror)
    registry = RunRegistry(global_limit=1)
    case_id = await registry.start_process(
        source,
        settings,
        submission_id="submission_admission_mirror_failure",
    )
    assert isinstance(case_id, str)
    handle = registry.handle(case_id)
    assert handle is not None and handle.task is not None
    with pytest.raises(InvoiceAgentsError) as raised:
        await asyncio.shield(handle.task)

    assert raised.value is admission_failure
    assert handle.state == "unresolved"
    assert case_id in registry._lifecycle_owners


@pytest.mark.asyncio
async def test_running_admission_transition_rejects_corrupt_batch_entry_without_repair(
    settings: Settings,
    tmp_path: Path,
) -> None:
    """The queued-to-running mirror must not overwrite contradictory batch state."""

    source = _write_distinct_invoice(tmp_path / "running-mirror.txt", "RUNNING-MIRROR")
    staged = await ui_runs._prepare_claimed_for_launch(source, settings)
    assert not isinstance(staged, CaseResult)
    store = WorkflowStore(settings)
    admission = store.claim_submission(
        "submission_running_mirror_corruption",
        "batch",
        (staged,),
        concurrency=1,
    )
    case_id = admission.cases[0].case_id
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE batch_entries SET state = 'failed' WHERE case_id = ?",
            (case_id,),
        )
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as corrupt:
        store.mark_admission_running(case_id)
    assert corrupt.value.stop_reason == "PERSISTED_SUBMISSION_INVALID"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        source_state = connection.execute(
            "SELECT state FROM source_run_claims WHERE case_id = ?", (case_id,)
        ).fetchone()[0]
        entry_state = connection.execute(
            "SELECT state FROM batch_entries WHERE case_id = ?", (case_id,)
        ).fetchone()[0]
    assert (source_state, entry_state) == ("queued", "failed")


@pytest.mark.asyncio
async def test_terminal_admission_transition_rejects_corrupt_batch_entry_without_repair(
    settings: Settings,
    tmp_path: Path,
) -> None:
    """The terminal mirror must not conceal a contradictory durable batch entry."""

    source = _write_distinct_invoice(tmp_path / "terminal-mirror.txt", "TERMINAL-MIRROR")
    staged = await ui_runs._prepare_claimed_for_launch(source, settings)
    assert not isinstance(staged, CaseResult)
    store = WorkflowStore(settings)
    admission = store.claim_submission(
        "submission_terminal_mirror_corruption",
        "batch",
        (staged,),
        concurrency=1,
    )
    admitted = admission.cases[0]
    assert admitted.claim is not None
    store.mark_admission_running(admitted.case_id)
    _finish_success(admitted.case_id, admitted.started_at, settings, admitted.claim)
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE batch_entries SET state = 'failed' WHERE case_id = ?",
            (admitted.case_id,),
        )
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as corrupt:
        store.mark_admission_terminal(admitted.case_id)
    assert corrupt.value.stop_reason == "PERSISTED_SUBMISSION_INVALID"
    with connect_database(settings.workflow_db, read_only=True) as connection:
        source_row = connection.execute(
            "SELECT state, released_at FROM source_run_claims WHERE case_id = ?",
            (admitted.case_id,),
        ).fetchone()
        entry_state = connection.execute(
            "SELECT state FROM batch_entries WHERE case_id = ?",
            (admitted.case_id,),
        ).fetchone()[0]
    assert source_row is not None and tuple(source_row) == ("running", None)
    assert entry_state == "failed"


@pytest.mark.parametrize("column", ("claimed_at", "released_at"))
@pytest.mark.asyncio
async def test_duplicate_batch_rejects_corrupt_source_claim_timestamps(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
) -> None:
    """A durable batch redirect must validate the complete current source claim."""

    source = _write_distinct_invoice(tmp_path / f"batch-{column}.txt", f"BATCH-{column}")
    calls: list[str] = []
    monkeypatch.setattr(ui_runs, "run_prepared_case", _successful_model(calls))
    registry = RunRegistry(global_limit=1)
    batch = await registry.start_batch(
        [source],
        settings,
        concurrency=1,
        submission_id=f"submission_batch_corrupt_{column}",
    )
    await _await_batch(batch)

    with connect_database(settings.workflow_db) as connection:
        connection.execute(f"UPDATE source_run_claims SET {column} = 'not-a-time'")
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as corrupt_claim:
        await RunRegistry(global_limit=1).start_batch(
            [source],
            settings,
            concurrency=1,
            submission_id=f"submission_batch_corrupt_{column}",
        )
    assert corrupt_claim.value.stop_reason == "PERSISTED_SUBMISSION_INVALID"
    assert calls == [batch.entries[0].case_id]


@pytest.mark.asyncio
async def test_idempotent_terminal_mirror_revalidates_the_complete_source_claim(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated terminal transition must not accept a corrupt released timestamp."""

    source = _write_distinct_invoice(tmp_path / "terminal-mirror.txt", "TERMINAL-MIRROR")
    calls: list[str] = []
    monkeypatch.setattr(ui_runs, "run_prepared_case", _successful_model(calls))
    registry = RunRegistry(global_limit=1)
    case_id = await registry.start_process(
        source,
        settings,
        submission_id="submission_terminal_mirror_validation",
    )
    assert isinstance(case_id, str)
    await _await_case(registry, case_id)

    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE source_run_claims SET released_at = NULL WHERE case_id = ?",
            (case_id,),
        )
        connection.commit()

    with pytest.raises(InvoiceAgentsError) as corrupt_claim:
        WorkflowStore(settings).mark_admission_terminal(case_id)
    assert corrupt_claim.value.stop_reason == "PERSISTED_SUBMISSION_INVALID"


@pytest.mark.asyncio
async def test_source_reuse_and_force_reprocess_are_durable_and_terminal_only(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force must not steal an active claim; ordinary admission must never spend twice."""

    source = _write_distinct_invoice(tmp_path / "force.txt", "FORCE-1")
    calls: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    monkeypatch.setattr(
        ui_runs,
        "run_prepared_case",
        _successful_model(calls, entered=entered, release=release),
    )
    registry = RunRegistry(global_limit=1)
    original = await registry.start_process(
        source,
        settings,
        submission_id="submission_force_original",
    )
    assert isinstance(original, str)
    await _await_event(entered)

    with pytest.raises(InvoiceAgentsError) as active_error:
        await RunRegistry(global_limit=1).start_process(
            source,
            settings,
            submission_id="submission_force_too_early",
            force_reprocess=True,
        )
    assert active_error.value.stop_reason == "SOURCE_RUN_NOT_TERMINAL"
    assert _case_count(settings) == 1

    release.set()
    await _await_case(registry, original)
    reused = await RunRegistry(global_limit=1).start_process(
        source,
        settings,
        submission_id="submission_ordinary_reuse",
    )
    assert reused == original
    assert calls == [original]
    assert _case_count(settings) == 1

    forced_registry = RunRegistry(global_limit=1)
    forced = await forced_registry.start_process(
        source,
        settings,
        submission_id="submission_force_terminal",
        force_reprocess=True,
    )
    assert isinstance(forced, str) and forced != original
    await _await_case(forced_registry, forced)
    assert calls == [original, forced]
    assert _case_count(settings) == 2

    historical = await RunRegistry(global_limit=1).start_process(
        source,
        settings,
        submission_id="submission_force_original",
    )
    assert historical == original
    assert calls == [original, forced]
    assert _case_count(settings) == 2


@pytest.mark.asyncio
async def test_completed_batch_and_entries_rehydrate_from_storage(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing RunRegistry must not make a completed batch URL disappear."""

    paths = [
        _write_distinct_invoice(tmp_path / f"durable-{index}.txt", f"DURABLE-{index}")
        for index in range(2)
    ]
    calls: list[str] = []
    prepare_calls: list[Path] = []
    real_prepare = ui_runs._prepare_claimed_for_launch

    async def observed_prepare(path: Path, selected: Settings):
        prepare_calls.append(path.resolve())
        return await real_prepare(path, selected)

    monkeypatch.setattr(ui_runs, "_prepare_claimed_for_launch", observed_prepare)
    monkeypatch.setattr(ui_runs, "run_prepared_case", _successful_model(calls))
    batch = await RunRegistry(global_limit=2).start_batch(
        paths,
        settings,
        concurrency=2,
        submission_id="submission_durable_batch",
    )
    await _await_batch(batch)
    prepared_before = _prepared_event_count(settings)
    all_events_before = _event_count(settings)
    cases_before = _case_count(settings)

    duplicate = await RunRegistry(global_limit=2).start_batch(
        paths,
        settings,
        concurrency=1,
        submission_id="submission_durable_batch",
    )
    assert duplicate.batch_id == batch.batch_id
    assert duplicate.task is None
    assert _case_count(settings) == cases_before == 2
    assert _prepared_event_count(settings) == prepared_before == 2
    assert _event_count(settings) == all_events_before
    assert prepare_calls == paths
    assert len(calls) == 2

    with connect_database(settings.workflow_db, read_only=True) as connection:
        batch_row = connection.execute(
            "SELECT state FROM batches WHERE batch_id = ?", (batch.batch_id,)
        ).fetchone()
        entry_rows = connection.execute(
            "SELECT state FROM batch_entries WHERE batch_id = ? ORDER BY position",
            (batch.batch_id,),
        ).fetchall()
    assert batch_row is not None and batch_row["state"] == "done"
    assert [row["state"] for row in entry_rows] == ["done", "done"]
    assert _submission_row(settings, "submission_durable_batch")["redirect_target"] == (
        f"/batches/{batch.batch_id}"
    )

    with TestClient(create_app(settings)) as restarted:
        response = restarted.get(f"/batches/{batch.batch_id}")
    assert response.status_code == 200
    assert all(path.name in response.text for path in paths)


@pytest.mark.asyncio
async def test_recovered_single_source_is_reconciled_before_ordinary_reuse(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 9 recovery must not leave Task 11 source admission permanently active."""

    source = _write_distinct_invoice(tmp_path / "recovered-single.txt", "RECOVERED-SINGLE")
    calls: list[str] = []

    async def interrupted_model(
        case_id: str,
        started_at: datetime,
        _selected: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        assert claim is not None
        calls.append(case_id)
        return CaseResult(
            case_id=case_id,
            source_id=WorkflowStore(settings).load_authoritative_case_source_id(claim),
            status=CaseStatus.SUCCEEDED,
            stop_reason="STALE_WORKER_RETURNED",
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    monkeypatch.setattr(ui_runs, "run_prepared_case", interrupted_model)
    first_registry = RunRegistry(global_limit=1)
    original = await first_registry.start_process(
        source,
        settings,
        submission_id="submission_recovered_single_original",
    )
    assert isinstance(original, str)
    original_handle = first_registry.handle(original)
    assert original_handle is not None and original_handle.task is not None
    with pytest.raises(InvoiceAgentsError) as unresolved:
        await asyncio.shield(original_handle.task)
    assert unresolved.value.stop_reason == "TERMINAL_DURABILITY_UNRESOLVED"

    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            (datetime(2000, 1, 1, tzinfo=UTC).isoformat(), original),
        )
        connection.commit()
    assert WorkflowStore(settings).recover_expired_executions() == [original]

    reused = await RunRegistry(global_limit=1).start_process(
        source,
        settings,
        submission_id="submission_recovered_single_reuse",
    )
    assert reused == original
    assert calls == [original]
    assert _case_count(settings) == 1
    with connect_database(settings.workflow_db, read_only=True) as connection:
        claim = connection.execute(
            "SELECT state, released_at FROM source_run_claims WHERE case_id = ?",
            (original,),
        ).fetchone()
    assert claim is not None and claim["state"] == "failed"
    assert claim["released_at"] is not None


@pytest.mark.asyncio
async def test_expired_batch_entry_rehydrates_as_failed_after_recovery(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restarted batch must not poll forever after Task 9 recovers its orphan."""

    source = _write_distinct_invoice(tmp_path / "interrupted.txt", "INTERRUPTED-1")

    async def interrupted_model(
        case_id: str,
        started_at: datetime,
        selected: Settings,
        *,
        claim: ExecutionClaim | None = None,
    ) -> CaseResult:
        del selected, claim
        return CaseResult(
            case_id=case_id,
            source_id=WorkflowStore(settings).load_case_source_id(case_id),
            status=CaseStatus.SUCCEEDED,
            stop_reason="STALE_WORKER_RETURNED",
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    monkeypatch.setattr(ui_runs, "run_prepared_case", interrupted_model)
    old_registry = RunRegistry(global_limit=1)
    batch = await old_registry.start_batch(
        [source],
        settings,
        concurrency=1,
        submission_id="submission_interrupted_batch",
    )
    original_task = batch.task
    assert original_task is not None
    with pytest.raises(InvoiceAgentsError) as unresolved:
        async with asyncio.timeout(10):
            await asyncio.shield(original_task)
    assert unresolved.value.stop_reason == "TERMINAL_DURABILITY_UNRESOLVED"
    turn_completed = asyncio.Event()
    asyncio.get_running_loop().call_soon(turn_completed.set)
    await _await_event(turn_completed)
    assert original_task.done()
    assert not any(
        batch.batch_id in task.get_name() and not task.done() for task in asyncio.all_tasks()
    )

    case_id = batch.entries[0].case_id
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ? WHERE case_id = ?",
            (datetime(2000, 1, 1, tzinfo=UTC).isoformat(), case_id),
        )
        connection.commit()

    with connect_database(settings.workflow_db, read_only=True) as connection:
        before_batch = connection.execute(
            "SELECT state FROM batches WHERE batch_id = ?", (batch.batch_id,)
        ).fetchone()
        before_entry = connection.execute(
            "SELECT state FROM batch_entries WHERE batch_id = ? AND case_id = ?",
            (batch.batch_id, case_id),
        ).fetchone()
    assert before_batch is not None and before_batch["state"] == "running"
    assert before_entry is not None and before_entry["state"] == "running"
    assert WorkflowStore(settings).load_result(case_id) is None

    restarted_app = create_app(settings)
    assert restarted_app.state.registry is not old_registry
    with TestClient(restarted_app) as restarted:
        response = restarted.get(f"/batches/{batch.batch_id}/rows")

    with connect_database(settings.workflow_db, read_only=True) as connection:
        batch_row = connection.execute(
            "SELECT state FROM batches WHERE batch_id = ?", (batch.batch_id,)
        ).fetchone()
        entry_row = connection.execute(
            "SELECT state FROM batch_entries WHERE batch_id = ? AND case_id = ?",
            (batch.batch_id, case_id),
        ).fetchone()
    result = WorkflowStore(settings).load_result(case_id)
    assert result is not None and result.stop_reason == "ORPHANED_EXECUTION"
    assert batch_row is not None and batch_row["state"] == "failed"
    assert entry_row is not None and entry_row["state"] == "failed"
    assert response.status_code == 286
    assert "ORPHANED_EXECUTION" in response.text
