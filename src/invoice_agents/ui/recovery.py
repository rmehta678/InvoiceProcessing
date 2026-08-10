"""Lifespan-owned recovery for abandoned durable execution claims."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Literal

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from invoice_agents.config import Settings
from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError

RecoveryState = Literal["created", "starting", "running", "failed", "stopping", "stopped"]


@dataclass(frozen=True, slots=True)
class RecoveryHealth:
    """Immutable recovery state safe to inspect from SSE worker threads."""

    state: RecoveryState
    version: int
    completed_scans: int
    last_successful_scan_at: datetime | None

    @property
    def available(self) -> bool:
        return self.state == "running"


class RecoveryCoordinator:
    """Own exactly one periodic recovery loop for one application lifespan."""

    def __init__(
        self,
        settings: Settings,
        *,
        scan_interval_seconds: float,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if scan_interval_seconds <= 0:
            raise ValueError("recovery scan interval must be positive")
        self._settings = settings
        self._scan_interval_seconds = scan_interval_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._state: RecoveryState = "created"
        self._version = 0
        self._completed_scans = 0
        self._last_successful_scan_at: datetime | None = None
        self._failure: BaseException | None = None
        self._state_lock = Lock()
        self._changed = asyncio.Condition()
        self._scan_requested = asyncio.Event()
        self._stop_requested = asyncio.Event()
        self._background_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> RecoveryState:
        return self.health().state

    @property
    def completed_scans(self) -> int:
        return self.health().completed_scans

    @property
    def background_task(self) -> asyncio.Task[None] | None:
        return self._background_task

    def health(self) -> RecoveryHealth:
        with self._state_lock:
            return RecoveryHealth(
                state=self._state,
                version=self._version,
                completed_scans=self._completed_scans,
                last_successful_scan_at=self._last_successful_scan_at,
            )

    async def _publish_state(
        self,
        state: RecoveryState,
        *,
        failure: BaseException | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        with self._state_lock:
            if completed_at is not None and self._state == "stopping":
                state = "stopping"
            self._state = state
            self._failure = failure
            if completed_at is not None:
                self._completed_scans += 1
                self._last_successful_scan_at = completed_at
            self._version += 1
        async with self._changed:
            self._changed.notify_all()

    async def _recover_once(self) -> None:
        scan_at = self._clock()
        if scan_at.tzinfo is None or scan_at.utcoffset() != UTC.utcoffset(scan_at):
            raise ValueError("recovery coordinator clock must be timezone-aware UTC")
        store = WorkflowStore(self._settings)
        await asyncio.to_thread(store.recover_expired_executions, now=scan_at)
        remaining = await asyncio.to_thread(
            store.unrecovered_execution_case_ids,
            checked_at=scan_at,
        )
        if remaining:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "execution recovery completed without terminalizing every eligible claim",
                stop_reason="EXECUTION_RECOVERY_INCOMPLETE",
                details={"unrecovered_case_ids": remaining},
            )
        await self._publish_state("running", completed_at=scan_at)

    async def start(self) -> None:
        if self.state == "stopped":
            task = self._background_task
            if task is not None and not task.done():
                raise RuntimeError("stopped recovery coordinator still owns a live task")
            with self._state_lock:
                self._state = "created"
                self._version = 0
                self._completed_scans = 0
                self._last_successful_scan_at = None
                self._failure = None
            self._changed = asyncio.Condition()
            self._scan_requested = asyncio.Event()
            self._stop_requested = asyncio.Event()
            self._background_task = None
        if self.state != "created":
            raise RuntimeError("recovery coordinator can only be started once")
        await self._publish_state("starting")
        try:
            await self._recover_once()
        except BaseException as exc:
            await self._publish_state("failed", failure=exc)
            raise
        self._background_task = asyncio.create_task(
            self._run(),
            name="invoice-ui-execution-recovery",
        )

    async def _run(self) -> None:
        try:
            while True:
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._scan_requested.wait(),
                        timeout=self._scan_interval_seconds,
                    )
                self._scan_requested.clear()
                if self._stop_requested.is_set():
                    return
                await self._recover_once()
        except asyncio.CancelledError as exc:
            if self._stop_requested.is_set():
                return
            failure = InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "execution recovery coordinator was cancelled outside application shutdown",
                stop_reason="EXECUTION_RECOVERY_CANCELLED",
            )
            await self._publish_state("failed", failure=failure)
            raise exc
        except BaseException as exc:
            await self._publish_state("failed", failure=exc)

    def request_scan(self) -> None:
        if self.state != "running":
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "execution recovery coordinator is unavailable",
                stop_reason="EXECUTION_RECOVERY_FAILED",
            )
        self._scan_requested.set()

    def _raise_failure(self) -> None:
        with self._state_lock:
            failure = self._failure
        if failure is not None:
            raise failure

    async def wait_for_scan(self, completed_after: int) -> int:
        async with self._changed:
            await self._changed.wait_for(
                lambda: self.completed_scans > completed_after or self.state != "running"
            )
        self._raise_failure()
        completed = self.completed_scans
        if completed <= completed_after:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "execution recovery coordinator stopped before the requested scan",
                stop_reason="EXECUTION_RECOVERY_FAILED",
            )
        return completed

    async def wait_for_change(self, version: int, *, timeout: float) -> RecoveryHealth:
        async with self._changed:
            try:
                async with asyncio.timeout(timeout):
                    await self._changed.wait_for(lambda: self.health().version != version)
            except TimeoutError:
                pass
        return self.health()

    async def close(self) -> None:
        state = self.state
        if state in {"created", "stopped"}:
            if state == "created":
                await self._publish_state("stopped")
            return
        if state == "starting":
            raise RuntimeError("recovery coordinator cannot close while start is incomplete")
        if state != "failed":
            await self._publish_state("stopping")
        self._stop_requested.set()
        self._scan_requested.set()
        task = self._background_task
        shutdown_cancellation: asyncio.CancelledError | None = None
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as cancellation:
                if not task.done():
                    shutdown_cancellation = cancellation
                    try:
                        await task
                    except asyncio.CancelledError:
                        if not task.cancelled():
                            raise
        self._raise_failure()
        await self._publish_state("stopped")
        if shutdown_cancellation is not None:
            raise shutdown_cancellation


class RecoveryHealthMiddleware:
    """Fail readiness after a runtime recovery-owner failure."""

    def __init__(self, app: ASGIApp, coordinator: RecoveryCoordinator) -> None:
        self.app = app
        self.coordinator = coordinator

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and not self.coordinator.health().available:
            response = PlainTextResponse(
                "Execution recovery unavailable",
                status_code=503,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
