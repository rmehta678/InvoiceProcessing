"""Lifespan-owned, fail-closed recovery for abandoned execution claims."""

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
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.isolated_process import ProcessCancellation
from invoice_agents.recovery_process import RecoveryProcessOutcome, run_recovery_process

RecoveryState = Literal["created", "starting", "running", "failed", "stopping", "stopped"]
DEFAULT_RECOVERY_WORKER_TIMEOUT_SECONDS = 120.0

type _ScanSignature = tuple[Path, Path]


@dataclass(slots=True)
class _ScanReservation:
    signature: _ScanSignature
    scan_at: datetime
    cancellation: ProcessCancellation
    worker: Future[RecoveryProcessOutcome]
    thread: Thread | None = None


_SCAN_RESERVATION: _ScanReservation | None = None
_RECOVERY_OWNERSHIP_POISONED = False
_RESERVATION_LOCK = Lock()


def _ownership_unresolved_error() -> InvoiceAgentsError:
    return InvoiceAgentsError(
        ErrorCategory.ORCHESTRATION,
        "execution recovery process ownership could not be proven clean",
        stop_reason="EXECUTION_RECOVERY_OWNERSHIP_UNRESOLVED",
    )


def _validate_process_outcome(value: object) -> RecoveryProcessOutcome:
    if type(value) is not RecoveryProcessOutcome:
        raise TypeError("recovery controller returned an invalid outcome type")
    return RecoveryProcessOutcome(
        value.acknowledged,
        value.error_category,
        value.stop_reason,
    )


def _outcome_error(outcome: RecoveryProcessOutcome) -> InvoiceAgentsError:
    validated = _validate_process_outcome(outcome)
    if validated.acknowledged or validated.error_category is None or validated.stop_reason is None:
        raise ValueError("acknowledged recovery does not carry a failure")
    return InvoiceAgentsError(
        validated.error_category,
        "the isolated execution recovery scan failed",
        stop_reason=validated.stop_reason,
    )


def _is_controller_outcome(outcome: RecoveryProcessOutcome, stop_reason: str) -> bool:
    return (
        outcome.error_category is ErrorCategory.ORCHESTRATION and outcome.stop_reason == stop_reason
    )


def _is_controller_error(error: InvoiceAgentsError, stop_reason: str) -> bool:
    return error.category is ErrorCategory.ORCHESTRATION and error.stop_reason == stop_reason


@dataclass(frozen=True, slots=True)
class RecoveryHealth:
    """Immutable recovery state safe to inspect from SSE worker threads."""

    state: RecoveryState
    version: int
    completed_scans: int
    last_successful_scan_at: datetime | None
    ownership_poisoned: bool

    @property
    def available(self) -> bool:
        return self.state == "running" and not self.ownership_poisoned


class RecoveryCoordinator:
    """Own exactly one periodic, isolated recovery loop for one app lifespan."""

    def __init__(
        self,
        settings: Settings,
        *,
        scan_interval_seconds: float,
        worker_timeout_seconds: float = DEFAULT_RECOVERY_WORKER_TIMEOUT_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if scan_interval_seconds <= 0:
            raise ValueError("recovery scan interval must be positive")
        if (
            type(worker_timeout_seconds) is not float
            or not isfinite(worker_timeout_seconds)
            or worker_timeout_seconds <= 0
        ):
            raise ValueError("recovery worker timeout must be a finite positive float")
        self._settings = settings
        self._scan_interval_seconds = scan_interval_seconds
        self._worker_timeout_seconds = worker_timeout_seconds
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
        self._active_scan_reservation: _ScanReservation | None = None

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
        with _RESERVATION_LOCK:
            poisoned = _RECOVERY_OWNERSHIP_POISONED
        with self._state_lock:
            return RecoveryHealth(
                state=self._state,
                version=self._version,
                completed_scans=self._completed_scans,
                last_successful_scan_at=self._last_successful_scan_at,
                ownership_poisoned=poisoned,
            )

    async def _publish_state(
        self,
        state: RecoveryState,
        *,
        failure: BaseException | None = None,
    ) -> None:
        with self._state_lock:
            self._state = state
            self._failure = failure
            self._version += 1
        async with self._changed:
            self._changed.notify_all()

    async def _publish_successful_scan(self, completed_at: datetime) -> None:
        with _RESERVATION_LOCK:
            if _RECOVERY_OWNERSHIP_POISONED:
                raise _ownership_unresolved_error()
            with self._state_lock:
                state: RecoveryState = "stopping" if self._state == "stopping" else "running"
                self._state = state
                self._failure = None
                self._completed_scans += 1
                self._last_successful_scan_at = completed_at
                self._version += 1
        async with self._changed:
            self._changed.notify_all()

    def _scan_signature(self) -> _ScanSignature:
        return (
            self._settings.workflow_db.resolve(),
            self._settings.inventory_db.resolve(),
        )

    def _reserve_scan(self, scan_at: datetime) -> tuple[_ScanReservation, bool]:
        global _RECOVERY_OWNERSHIP_POISONED, _SCAN_RESERVATION

        with _RESERVATION_LOCK:
            if _RECOVERY_OWNERSHIP_POISONED:
                raise _ownership_unresolved_error()
            if _SCAN_RESERVATION is not None:
                return _SCAN_RESERVATION, False

            cancellation = ProcessCancellation()
            future: Future[RecoveryProcessOutcome] = Future()
            candidate = _ScanReservation(
                signature=self._scan_signature(),
                scan_at=scan_at,
                cancellation=cancellation,
                worker=future,
            )

            def run() -> None:
                global _RECOVERY_OWNERSHIP_POISONED

                try:
                    outcome = _validate_process_outcome(
                        run_recovery_process(
                            settings=self._settings,
                            scan_at=scan_at,
                            timeout_seconds=self._worker_timeout_seconds,
                            cancel_requested=cancellation,
                        )
                    )
                except BaseException as exc:
                    with _RESERVATION_LOCK:
                        _RECOVERY_OWNERSHIP_POISONED = True
                    future.set_exception(exc)
                    return
                with _RESERVATION_LOCK:
                    if _is_controller_outcome(
                        outcome,
                        "EXECUTION_RECOVERY_WORKER_CLEANUP_FAILED",
                    ):
                        _RECOVERY_OWNERSHIP_POISONED = True
                future.set_result(outcome)

            thread = Thread(
                target=run,
                name="invoice-ui-execution-recovery-scan",
                daemon=False,
            )
            candidate.thread = thread
            _SCAN_RESERVATION = candidate
            try:
                thread.start()
            except BaseException:
                try:
                    possibly_started = thread.ident is not None or thread.is_alive()
                except BaseException:
                    possibly_started = True
                if possibly_started:
                    cancellation.set()
                    _RECOVERY_OWNERSHIP_POISONED = True
                else:
                    _SCAN_RESERVATION = None
                raise
            return candidate, True

    @staticmethod
    def _retire_completed_reservation(reservation: _ScanReservation) -> None:
        global _RECOVERY_OWNERSHIP_POISONED, _SCAN_RESERVATION

        thread = reservation.thread
        if thread is None or not reservation.worker.done():
            with _RESERVATION_LOCK:
                _RECOVERY_OWNERSHIP_POISONED = True
            raise _ownership_unresolved_error()
        try:
            thread.join()
        except BaseException:
            with _RESERVATION_LOCK:
                _RECOVERY_OWNERSHIP_POISONED = True
            raise _ownership_unresolved_error() from None
        try:
            thread_is_alive = thread.is_alive()
        except BaseException:
            with _RESERVATION_LOCK:
                _RECOVERY_OWNERSHIP_POISONED = True
            raise _ownership_unresolved_error() from None
        if thread_is_alive:
            with _RESERVATION_LOCK:
                _RECOVERY_OWNERSHIP_POISONED = True
            raise _ownership_unresolved_error()
        with _RESERVATION_LOCK:
            if not _RECOVERY_OWNERSHIP_POISONED and _SCAN_RESERVATION is reservation:
                _SCAN_RESERVATION = None

    def _set_active_reservation(self, reservation: _ScanReservation | None) -> None:
        with self._state_lock:
            self._active_scan_reservation = reservation

    def _cancel_active_reservation(self) -> None:
        with self._state_lock:
            reservation = self._active_scan_reservation
        if reservation is not None and not reservation.worker.done():
            reservation.cancellation.set()

    async def _wait_for_owned_scan(
        self,
        reservation: _ScanReservation,
    ) -> RecoveryProcessOutcome:
        wrapped = asyncio.wrap_future(reservation.worker)
        cancellation: asyncio.CancelledError | None = None
        while not wrapped.done():
            try:
                await asyncio.shield(wrapped)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                reservation.cancellation.set()
            except BaseException:
                break
        try:
            outcome = wrapped.result()
        finally:
            self._retire_completed_reservation(reservation)
            self._set_active_reservation(None)
        if not outcome.acknowledged:
            if cancellation is not None and _is_controller_outcome(
                outcome,
                "EXECUTION_RECOVERY_WORKER_CANCELLED",
            ):
                raise cancellation
            raise _outcome_error(outcome)
        if cancellation is not None:
            raise cancellation
        return outcome

    async def _wait_for_shared_scan(
        self,
        reservation: _ScanReservation,
    ) -> RecoveryProcessOutcome:
        try:
            outcome = await asyncio.shield(asyncio.wrap_future(reservation.worker))
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._retire_completed_reservation(reservation)
            with _RESERVATION_LOCK:
                poisoned = _RECOVERY_OWNERSHIP_POISONED
            if poisoned:
                raise _ownership_unresolved_error() from None
            raise
        self._retire_completed_reservation(reservation)
        return outcome

    async def _recover_once(self) -> None:
        signature = self._scan_signature()
        while True:
            scan_at = self._clock()
            if type(scan_at) is not datetime or scan_at.tzinfo is not UTC:
                raise ValueError("recovery coordinator clock must be canonical UTC")
            reservation, owns_worker = self._reserve_scan(scan_at)
            if owns_worker:
                self._set_active_reservation(reservation)
                try:
                    outcome = await self._wait_for_owned_scan(reservation)
                except InvoiceAgentsError as exc:
                    if self._stop_requested.is_set() and _is_controller_error(
                        exc,
                        "EXECUTION_RECOVERY_WORKER_CANCELLED",
                    ):
                        return
                    raise
                if not outcome.acknowledged:
                    raise _outcome_error(outcome)
                await self._publish_successful_scan(reservation.scan_at)
                return

            outcome = await self._wait_for_shared_scan(reservation)
            if not outcome.acknowledged:
                if _is_controller_outcome(
                    outcome,
                    "EXECUTION_RECOVERY_WORKER_CLEANUP_FAILED",
                ):
                    raise _ownership_unresolved_error()
                raise _outcome_error(outcome)
            if reservation.signature != signature:
                continue
            # Another coordinator's successful pass does not authorize this
            # coordinator. Wait for release, then run a fresh pass of our own.

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
                self._active_scan_reservation = None
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

    async def _drain_background_task(
        self,
        task: asyncio.Task[None],
        cancellation: asyncio.CancelledError | None,
    ) -> asyncio.CancelledError | None:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                self._cancel_active_reservation()
        return cancellation

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
            except asyncio.CancelledError as exc:
                shutdown_cancellation = exc
        self._stop_requested.set()
        self._scan_requested.set()
        self._cancel_active_reservation()
        task = self._background_task
        if task is not None and not task.done():
            shutdown_cancellation = await self._drain_background_task(
                task,
                shutdown_cancellation,
            )
        self._raise_failure()
        try:
            await self._publish_state("stopped")
        except asyncio.CancelledError as exc:
            if shutdown_cancellation is None:
                shutdown_cancellation = exc
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
