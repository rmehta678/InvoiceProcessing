"""Lifespan-owned recovery for abandoned durable execution claims."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from threading import Lock, Thread
from typing import Literal

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from invoice_agents.config import Settings
from invoice_agents.db.store import WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError

RecoveryState = Literal["created", "starting", "running", "failed", "stopping", "stopped"]
DEFAULT_RECOVERY_SHUTDOWN_TIMEOUT_SECONDS = 5.0

type _ScanOutcome = Literal["succeeded", "failed", "quarantined"]
type _ScanSignature = tuple[Path, Path]


@dataclass(slots=True)
class _ScanReservation:
    signature: _ScanSignature
    scan_at: datetime
    worker: Future[None]
    settled: Future[_ScanOutcome]
    quarantined: bool = False


_SCAN_RESERVATION: _ScanReservation | None = None
_QUARANTINED_BACKGROUND_TASK: asyncio.Task[None] | None = None
_QUARANTINE_LOCK = Lock()


def _ownership_unresolved_error() -> InvoiceAgentsError:
    return InvoiceAgentsError(
        ErrorCategory.ORCHESTRATION,
        "a prior execution recovery worker still owns unresolved process resources",
        stop_reason="EXECUTION_RECOVERY_OWNERSHIP_UNRESOLVED",
    )


def _finish_scan_reservation(reservation: _ScanReservation) -> None:
    global _SCAN_RESERVATION
    failure = reservation.worker.exception()
    with _QUARANTINE_LOCK:
        if _SCAN_RESERVATION is reservation:
            _SCAN_RESERVATION = None
        if not reservation.settled.done():
            reservation.settled.set_result("failed" if failure is not None else "succeeded")


def _quarantine_scan(future: Future[None]) -> None:
    with _QUARANTINE_LOCK:
        reservation = _SCAN_RESERVATION
        if reservation is None or reservation.worker is not future or reservation.worker.done():
            return
        reservation.quarantined = True
        if not reservation.settled.done():
            reservation.settled.set_result("quarantined")


def _consume_quarantined_background_task(task: asyncio.Task[None]) -> None:
    global _QUARANTINED_BACKGROUND_TASK
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    finally:
        with _QUARANTINE_LOCK:
            if _QUARANTINED_BACKGROUND_TASK is task:
                _QUARANTINED_BACKGROUND_TASK = None


def _quarantine_background_task(task: asyncio.Task[None]) -> None:
    global _QUARANTINED_BACKGROUND_TASK
    with _QUARANTINE_LOCK:
        retained = _QUARANTINED_BACKGROUND_TASK
        if retained is task:
            return
        if retained is not None and not retained.done():
            raise _ownership_unresolved_error()
        _QUARANTINED_BACKGROUND_TASK = task
    task.add_done_callback(_consume_quarantined_background_task)


@dataclass(frozen=True, slots=True)
class RecoveryHealth:
    """Immutable recovery state safe to inspect from SSE worker threads."""

    state: RecoveryState
    version: int
    completed_scans: int
    last_successful_scan_at: datetime | None
    quarantined_scans: int

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
        shutdown_timeout_seconds: float = DEFAULT_RECOVERY_SHUTDOWN_TIMEOUT_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if scan_interval_seconds <= 0:
            raise ValueError("recovery scan interval must be positive")
        if not isfinite(shutdown_timeout_seconds) or shutdown_timeout_seconds <= 0:
            raise ValueError("recovery shutdown timeout must be finite and positive")
        self._settings = settings
        self._scan_interval_seconds = scan_interval_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
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
        self._active_scan_future: Future[None] | None = None

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
                quarantined_scans=int(
                    self._active_scan_future is not None
                    and not self._active_scan_future.done()
                    and self._state in {"failed", "stopped"}
                ),
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

    def _recover_in_thread(self, scan_at: datetime) -> None:
        store = WorkflowStore(self._settings)
        store.recover_expired_executions(now=scan_at)
        remaining = store.unrecovered_execution_case_ids(checked_at=scan_at)
        if remaining:
            raise InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "execution recovery completed without terminalizing every eligible claim",
                stop_reason="EXECUTION_RECOVERY_INCOMPLETE",
                details={"unrecovered_case_ids": remaining},
            )

    def _scan_signature(self) -> _ScanSignature:
        return (
            self._settings.workflow_db.resolve(),
            self._settings.inventory_db.resolve(),
        )

    def _reserve_scan(self, scan_at: datetime) -> tuple[_ScanReservation, bool]:
        global _QUARANTINED_BACKGROUND_TASK, _SCAN_RESERVATION

        future: Future[None] = Future()
        candidate = _ScanReservation(
            signature=self._scan_signature(),
            scan_at=scan_at,
            worker=future,
            settled=Future(),
        )

        def run() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                self._recover_in_thread(scan_at)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)

        def finish(_future: Future[None]) -> None:
            _finish_scan_reservation(candidate)

        thread = Thread(
            target=run,
            name="invoice-ui-execution-recovery-scan",
            daemon=True,
        )
        with _QUARANTINE_LOCK:
            background = _QUARANTINED_BACKGROUND_TASK
            if background is not None:
                if not background.done():
                    raise _ownership_unresolved_error()
                _QUARANTINED_BACKGROUND_TASK = None

            reservation = _SCAN_RESERVATION
            if reservation is not None and reservation.worker.done():
                failure = reservation.worker.exception()
                if not reservation.settled.done():
                    reservation.settled.set_result("failed" if failure is not None else "succeeded")
                _SCAN_RESERVATION = None
                reservation = None
            if reservation is not None:
                if reservation.quarantined:
                    raise _ownership_unresolved_error()
                return reservation, False

            _SCAN_RESERVATION = candidate
            self._active_scan_future = future
            future.add_done_callback(finish)
            try:
                thread.start()
            except BaseException:
                _SCAN_RESERVATION = None
                self._active_scan_future = None
                candidate.settled.set_result("failed")
                raise
        return candidate, True

    async def _recover_once(self) -> None:
        signature = self._scan_signature()
        while True:
            scan_at = self._clock()
            if scan_at.tzinfo is None or scan_at.utcoffset() != UTC.utcoffset(scan_at):
                raise ValueError("recovery coordinator clock must be timezone-aware UTC")
            reservation, owns_worker = self._reserve_scan(scan_at)
            if owns_worker:
                try:
                    await asyncio.shield(asyncio.wrap_future(reservation.worker))
                except asyncio.CancelledError:
                    if not reservation.worker.done():
                        _quarantine_scan(reservation.worker)
                    raise
                finally:
                    if reservation.worker.done() and self._active_scan_future is reservation.worker:
                        self._active_scan_future = None
                await self._publish_state("running", completed_at=reservation.scan_at)
                return

            outcome = await asyncio.shield(asyncio.wrap_future(reservation.settled))
            if outcome == "quarantined":
                raise _ownership_unresolved_error()
            if reservation.signature != signature:
                continue
            if outcome == "failed":
                raise InvoiceAgentsError(
                    ErrorCategory.ORCHESTRATION,
                    "the shared execution recovery scan failed",
                    stop_reason="EXECUTION_RECOVERY_FAILED",
                )
            # A successful reservation proves only its owner's scan. Run this
            # coordinator's own fresh scan once the process-wide slot is free.

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
            self._active_scan_future = None
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
        shutdown_cancellation: asyncio.CancelledError | None = None
        if state != "failed":
            try:
                await self._publish_state("stopping")
            except asyncio.CancelledError as cancellation:
                shutdown_cancellation = cancellation
        self._stop_requested.set()
        self._scan_requested.set()
        task = self._background_task
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._shutdown_timeout_seconds
        if task is not None and not task.done():
            shutdown_cancellation = await self._drain_background_task(
                task,
                deadline=deadline,
                cancellation=shutdown_cancellation,
            )
        if task is not None and not task.done():
            active_scan = self._active_scan_future
            if active_scan is not None and not active_scan.done():
                _quarantine_scan(active_scan)
            task.cancel()
            cancellation_deadline = loop.time() + self._shutdown_timeout_seconds
            shutdown_cancellation = await self._drain_background_task(
                task,
                deadline=cancellation_deadline,
                cancellation=shutdown_cancellation,
            )
            if not task.done():
                _quarantine_background_task(task)
            failure = InvoiceAgentsError(
                ErrorCategory.ORCHESTRATION,
                "execution recovery scan did not drain before the shutdown deadline",
                stop_reason="EXECUTION_RECOVERY_DRAIN_TIMEOUT",
            )
            with suppress(asyncio.CancelledError):
                await self._publish_state("failed", failure=failure)
            raise failure
        self._raise_failure()
        try:
            await self._publish_state("stopped")
        except asyncio.CancelledError as cancellation:
            if shutdown_cancellation is None:
                shutdown_cancellation = cancellation
        if shutdown_cancellation is not None:
            raise shutdown_cancellation

    async def _drain_background_task(
        self,
        task: asyncio.Task[None],
        *,
        deadline: float,
        cancellation: asyncio.CancelledError | None,
    ) -> asyncio.CancelledError | None:
        loop = asyncio.get_running_loop()
        while not task.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                async with asyncio.timeout(remaining):
                    await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
            except TimeoutError:
                break
        return cancellation


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
