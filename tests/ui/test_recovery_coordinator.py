"""Application-owned execution recovery and observational SSE contracts."""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Any, cast

import pytest
from fastapi import FastAPI

from invoice_agents import orchestration
from invoice_agents.config import Settings
from invoice_agents.db.core import connect_database
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import InvoiceAgentsError
from invoice_agents.models import CaseStatus
from invoice_agents.ui import server as ui_server
from invoice_agents.ui import sse
from invoice_agents.ui.recovery import RecoveryCoordinator
from invoice_agents.ui.runs import RunRegistry


def _prepared_case(invoice_dir: Path, settings: Settings) -> tuple[str, datetime]:
    prepared = orchestration.prepare_case(invoice_dir / "invoice_1001.txt", settings)
    assert isinstance(prepared, tuple)
    return prepared


def _expire_claim(
    invoice_dir: Path,
    settings: Settings,
) -> tuple[str, ExecutionClaim, str]:
    case_id, _started_at = _prepared_case(invoice_dir, settings)
    claim = WorkflowStore(settings).claim_case_execution(
        case_id,
        frozenset({CaseStatus.INCOMPLETE}),
        lease_seconds=60,
    )
    expired_at = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with connect_database(settings.workflow_db) as connection:
        connection.execute(
            "UPDATE cases SET lease_expires_at = ?, updated_at = ? WHERE case_id = ?",
            (expired_at, expired_at, case_id),
        )
        connection.commit()
    return case_id, claim, expired_at


def _authority_and_recovery_count(settings: Settings, case_id: str) -> tuple[Any, ...]:
    with connect_database(settings.workflow_db, read_only=True) as connection:
        row = connection.execute(
            "SELECT status, stop_reason, result_json, finished_at, execution_token, "
            "execution_generation, execution_state, lease_expires_at FROM cases "
            "WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE case_id = ? "
            "AND event_type = 'case.execution_recovered'",
            (case_id,),
        ).fetchone()[0]
    return (*tuple(row), count)


def _coordinator(app: FastAPI) -> RecoveryCoordinator:
    coordinator = getattr(app.state, "recovery_coordinator", None)
    assert coordinator is not None, "the FastAPI lifespan must own execution recovery"
    return cast(RecoveryCoordinator, coordinator)


async def _get_once(app: FastAPI, path: str) -> list[dict[str, Any]]:
    delivered = False
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", b"localhost:8787")],
        "client": ("127.0.0.1", 41_000),
        "server": ("127.0.0.1", 8_787),
        "app": app,
    }
    await app(scope, receive, send)
    return sent


@pytest.mark.asyncio
async def test_lifespan_coordinator_recovers_post_startup_expiry_and_sse_only_observes(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    app = ui_server.create_app(settings)
    coordinator = _coordinator(app)
    background_task: asyncio.Task[None] | None = None

    async with app.router.lifespan_context(app):
        background_task = coordinator.background_task
        assert background_task is not None and not background_task.done()
        expired_case, expired_claim, _expired_at = _expire_claim(invoice_dir, settings)
        active_case, _ = _prepared_case(invoice_dir, settings)
        active_claim = WorkflowStore(settings).claim_case_execution(
            active_case,
            frozenset({CaseStatus.INCOMPLETE}),
            lease_seconds=60,
        )
        active_before = _authority_and_recovery_count(settings, active_case)
        stream = sse.case_event_stream(
            settings.workflow_db,
            expired_case,
            RunRegistry(),
            after_seq=1_000_000,
            settings=settings,
            recovery_coordinator=coordinator,
        )
        terminal_event = asyncio.create_task(anext(stream))

        completed_before = coordinator.completed_scans
        coordinator.request_scan()
        await asyncio.wait_for(
            coordinator.wait_for_scan(completed_before),
            timeout=1,
        )
        event = await asyncio.wait_for(terminal_event, timeout=1)
        await stream.aclose()

        recovered = WorkflowStore(settings).load_result(expired_case)
        assert recovered is not None
        assert recovered.stop_reason == "ORPHANED_EXECUTION"
        recovered_state = _authority_and_recovery_count(settings, expired_case)
        assert recovered_state[5:] == (
            expired_claim.generation + 1,
            "FINISHED",
            None,
            1,
        )
        assert b"event: terminal" in event.encode()
        assert b"ORPHANED_EXECUTION" in event.encode()
        assert _authority_and_recovery_count(settings, active_case) == active_before
        assert WorkflowStore(settings).has_valid_execution_lease(active_case)
        assert active_before[4:8] == (
            active_claim.token,
            active_claim.generation,
            "RUNNING",
            active_claim.expires_at.isoformat(),
        )

    assert background_task is not None and background_task.done()
    assert coordinator.state == "stopped"


@pytest.mark.asyncio
async def test_two_application_coordinators_recover_one_generation_and_one_audit(
    invoice_dir: Path,
    settings: Settings,
) -> None:
    first_app = ui_server.create_app(settings)
    second_app = ui_server.create_app(settings)
    first = _coordinator(first_app)
    second = _coordinator(second_app)

    async with AsyncExitStack() as stack:
        await stack.enter_async_context(first_app.router.lifespan_context(first_app))
        await stack.enter_async_context(second_app.router.lifespan_context(second_app))
        case_id, claim, _expired_at = _expire_claim(invoice_dir, settings)
        first_before = first.completed_scans
        second_before = second.completed_scans

        first.request_scan()
        second.request_scan()
        await asyncio.wait_for(
            asyncio.gather(
                first.wait_for_scan(first_before),
                second.wait_for_scan(second_before),
            ),
            timeout=2,
        )

        recovered = WorkflowStore(settings).load_result(case_id)
        assert recovered is not None and recovered.stop_reason == "ORPHANED_EXECUTION"
        state = _authority_and_recovery_count(settings, case_id)
        assert state[5:] == (claim.generation + 1, "FINISHED", None, 1)


@pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost:8787"])
@pytest.mark.asyncio
async def test_sse_get_is_observational_for_hostile_and_same_origin_requests(
    origin: str,
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ui_server, "RECOVERY_SCAN_INTERVAL_SECONDS", 3_600, raising=False)
    app = ui_server.create_app(settings)
    coordinator = _coordinator(app)

    async with app.router.lifespan_context(app):
        case_id, _claim, _expired_at = _expire_claim(invoice_dir, settings)
        before = _authority_and_recovery_count(settings, case_id)
        original_snapshot = WorkflowStore.load_case_execution_snapshot
        snapshot_observed = asyncio.Event()
        loop = asyncio.get_running_loop()

        def observe_snapshot(
            self: WorkflowStore,
            selected_case_id: str,
            *,
            checked_at: datetime | None = None,
        ) -> object:
            snapshot = original_snapshot(self, selected_case_id, checked_at=checked_at)
            loop.call_soon_threadsafe(snapshot_observed.set)
            return snapshot

        monkeypatch.setattr(WorkflowStore, "load_case_execution_snapshot", observe_snapshot)
        disconnected = asyncio.Event()
        request_delivered = False
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            nonlocal request_delivered
            if not request_delivered:
                request_delivered = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": f"/cases/{case_id}/events",
            "raw_path": f"/cases/{case_id}/events".encode(),
            "query_string": b"after=1000000",
            "headers": [
                (b"host", b"localhost:8787"),
                (b"origin", origin.encode()),
            ],
            "client": ("127.0.0.1", 41_000),
            "server": ("127.0.0.1", 8_787),
            "app": app,
        }
        request = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(snapshot_observed.wait(), timeout=1)

        assert _authority_and_recovery_count(settings, case_id) == before
        disconnected.set()
        await asyncio.wait_for(request, timeout=1)
        assert (
            next(message for message in sent if message["type"] == "http.response.start")["status"]
            == 200
        )
        assert _authority_and_recovery_count(settings, case_id) == before
        assert coordinator.state == "running"


@pytest.mark.parametrize("silent_noop", [False, True])
@pytest.mark.asyncio
async def test_runtime_recovery_failure_is_health_failing_and_never_fakes_terminal_state(
    silent_noop: bool,
    invoice_dir: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = ui_server.create_app(settings)
    coordinator = _coordinator(app)
    background_task: asyncio.Task[None] | None = None

    with pytest.raises((RuntimeError, InvoiceAgentsError)):
        async with app.router.lifespan_context(app):
            background_task = coordinator.background_task
            case_id, claim, expired_at = _expire_claim(invoice_dir, settings)
            before = _authority_and_recovery_count(settings, case_id)
            stream = sse.case_event_stream(
                settings.workflow_db,
                case_id,
                RunRegistry(),
                after_seq=1_000_000,
                settings=settings,
                recovery_coordinator=coordinator,
            )
            stream_event = asyncio.create_task(anext(stream))

            def failed_recovery(
                _self: WorkflowStore,
                **_kwargs: object,
            ) -> list[str]:
                if silent_noop:
                    return []
                raise RuntimeError("runtime recovery sentinel")

            monkeypatch.setattr(WorkflowStore, "recover_expired_executions", failed_recovery)
            completed_before = coordinator.completed_scans
            coordinator.request_scan()
            with pytest.raises((RuntimeError, InvoiceAgentsError)):
                await asyncio.wait_for(
                    coordinator.wait_for_scan(completed_before),
                    timeout=1,
                )
            unavailable_event = await asyncio.wait_for(stream_event, timeout=1)
            await stream.aclose()
            assert b"event: error" in unavailable_event.encode()
            assert b"EXECUTION_RECOVERY_FAILED" in unavailable_event.encode()

            payload = sse.terminal_payload(
                settings.workflow_db,
                case_id,
                RunRegistry(),
                settings,
                recovery_coordinator=coordinator,
            )
            assert payload is not None
            assert payload["status"] == "UNAVAILABLE"
            assert payload["recovery_verified"] is False
            assert "runtime recovery sentinel" not in json.dumps(payload)
            assert _authority_and_recovery_count(settings, case_id) == before
            assert WorkflowStore(settings).load_result(case_id) is None
            assert before[4:] == (
                claim.token,
                claim.generation,
                "RUNNING",
                expired_at,
                0,
            )
            assert coordinator.state == "failed"
            response = await _get_once(app, "/")
            start = next(
                message for message in response if message["type"] == "http.response.start"
            )
            assert start["status"] == 503
            assert b"Execution recovery unavailable" in b"".join(
                message.get("body", b"")
                for message in response
                if message["type"] == "http.response.body"
            )

    assert background_task is not None and background_task.done()
    assert coordinator.state == "failed"


@pytest.mark.asyncio
async def test_shutdown_cancellation_drains_the_owned_scan_before_propagating(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = ui_server.create_app(settings)
    coordinator = _coordinator(app)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    entered = Event()
    release = Event()
    original_recovery = WorkflowStore.recover_expired_executions
    background_task = coordinator.background_task

    def blocking_recovery(self: WorkflowStore, **kwargs: object) -> list[str]:
        entered.set()
        release.wait()
        return original_recovery(self, **kwargs)

    monkeypatch.setattr(WorkflowStore, "recover_expired_executions", blocking_recovery)
    try:
        coordinator.request_scan()
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1)
        version = coordinator.health().version
        close_task = asyncio.create_task(coordinator.close())
        health = await coordinator.wait_for_change(version, timeout=1)
        assert health.state == "stopping"

        close_task.cancel()
        await asyncio.sleep(0)
        assert not close_task.done(), "shutdown cancellation abandoned the owned recovery scan"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        assert background_task is not None and background_task.done()
        assert coordinator.state == "stopped"
    finally:
        release.set()
        if coordinator.state != "stopped":
            await coordinator.close()
        await lifespan.__aexit__(None, None, None)
