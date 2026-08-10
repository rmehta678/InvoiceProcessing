"""Background run registry over the existing orchestration services.

Runs are keyed by case ID and reuse the same bounded-concurrency semantics as
``process_batch``. The registry only knows whether a task is in flight; case
status and execution authority always come from the workflow database claim
acquired by the orchestration entrypoint used by both CLI and UI callers.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from invoice_agents.config import Settings
from invoice_agents.db.store import ExecutionClaim, WorkflowStore
from invoice_agents.errors import ErrorCategory, InvoiceAgentsError
from invoice_agents.models import CaseResult
from invoice_agents.orchestration import (
    _DURABILITY_PRECEDENCE_STOPS,
    DURABILITY_DEADLINE_SECONDS,
    TERMINAL_WORKER_CLEANUP_GRACE_SECONDS,
    _await_task_despite_cancellation,
    _durably_cancel_unstarted_claim,
    _durably_cancel_unstarted_claims,
    _inspect_claim_durability,
    _select_drained_failure,
    claim_resumable_case,
    prepare_claimed_invoice_async,
    resume_case,
    run_prepared_case,
    validate_case_concurrency,
)


@dataclass(slots=True)
class RunHandle:
    """One in-flight or completed background run for a case."""

    case_id: str
    kind: str  # "process" | "resume" | "batch"
    state: str  # "queued" | "running" | "done" | "unresolved"
    started_at: datetime
    task: asyncio.Task[CaseResult] | None = None
    error: str | None = None


@dataclass(slots=True)
class BatchEntry:
    """One submitted file in a batch; prepare-time failures keep their result."""

    path: Path
    case_id: str
    prepared_started_at: datetime | None
    result: CaseResult | None = None

    @property
    def prepared(self) -> bool:
        return self.prepared_started_at is not None


@dataclass(slots=True)
class BatchState:
    batch_id: str
    created_at: datetime
    concurrency: int
    entries: list[BatchEntry] = field(default_factory=list)
    task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


@dataclass(slots=True, repr=False)
class _LifecycleOwner:
    """Private execution authority retained only while durability is unresolved."""

    claim: ExecutionClaim = field(repr=False)
    started_at: datetime


def _now() -> datetime:
    return datetime.now(UTC)


def _stable_run_error(exc: BaseException) -> str:
    """Map arbitrary child failures to one parent-owned, non-secret registry code."""

    if isinstance(exc, InvoiceAgentsError):
        return f"BACKGROUND_{exc.category.value}_ERROR"
    if isinstance(exc, asyncio.CancelledError):
        return "BACKGROUND_CANCELLED"
    return "UNEXPECTED_RUNTIME_ERROR"


async def _prepare_claimed_for_launch(
    path: Path, settings: Settings
) -> tuple[str, datetime, ExecutionClaim] | CaseResult:
    """Prepare through the shared session-owned, cancellation-safe boundary."""

    return await prepare_claimed_invoice_async(path, settings)


class RunRegistry:
    """Track in-flight asyncio runs; storage remains the source of truth."""

    def __init__(self) -> None:
        self._runs: dict[str, RunHandle] = {}
        self._sources: dict[str, str] = {}
        self._batches: dict[str, BatchState] = {}
        self._lifecycle_owners: dict[str, _LifecycleOwner] = {}

    def _install_lifecycle_owner(
        self,
        case_id: str,
        started_at: datetime,
        claim: ExecutionClaim,
    ) -> _LifecycleOwner:
        if case_id in self._lifecycle_owners:
            raise InvoiceAgentsError(
                category=ErrorCategory.ORCHESTRATION,
                message="case already has a private lifecycle owner",
                case_id=case_id,
                stop_reason="CASE_ALREADY_CLAIMED",
            )
        owner = _LifecycleOwner(claim=claim, started_at=started_at)
        self._lifecycle_owners[case_id] = owner
        return owner

    def _drop_lifecycle_owner(self, case_id: str, owner: _LifecycleOwner) -> None:
        if self._lifecycle_owners.get(case_id) is owner:
            del self._lifecycle_owners[case_id]

    def handle(self, case_id: str) -> RunHandle | None:
        return self._runs.get(case_id)

    def is_running(self, case_id: str) -> bool:
        handle = self._runs.get(case_id)
        return handle is not None and handle.state in {"queued", "running"}

    def run_state(self, case_id: str) -> str | None:
        handle = self._runs.get(case_id)
        return handle.state if handle else None

    def run_error(self, case_id: str) -> str | None:
        handle = self._runs.get(case_id)
        return handle.error if handle else None

    def running_case_for_source(self, path: Path) -> str | None:
        """The case currently running for this source file, if any."""

        case_id = self._sources.get(str(path.resolve()))
        if case_id is not None and self.is_running(case_id):
            return case_id
        return None

    def batch(self, batch_id: str) -> BatchState | None:
        return self._batches.get(batch_id)

    def _finish(self, handle: RunHandle, source_key: str | None) -> None:
        # The lifecycle owner, not this callback, decides whether terminal
        # durability was proved.  A callback can run after same-tick
        # cancellation, so treating task completion itself as success would
        # orphan the already-issued execution claim.
        if handle.state in {"queued", "running"}:
            handle.state = "unresolved"
        if source_key is not None and self._sources.get(source_key) == handle.case_id:
            del self._sources[source_key]
        task = handle.task
        if task is not None and task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                handle.error = _stable_run_error(exc)
                exc.__traceback__ = None
                exc.__cause__ = None
                exc.__context__ = None
        if task is not None and task.done():
            handle.task = None

    def _finish_batch(self, batch: BatchState) -> None:
        """Retain only stable batch display state after its owner has completed."""

        task = batch.task
        if task is not None and task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                exc.__traceback__ = None
                exc.__cause__ = None
                exc.__context__ = None
        if task is not None and task.done():
            batch.task = None

    async def start_process(self, path: Path, settings: Settings) -> str | CaseResult:
        """Prepare and launch one case; terminal prepare failures are returned as-is."""

        prepared = await _prepare_claimed_for_launch(path, settings)
        if isinstance(prepared, CaseResult):
            return prepared
        case_id, started_at, claim = prepared
        await self._launch(
            case_id,
            "process",
            run_prepared_case(case_id, started_at, settings, claim=claim),
            claim=claim,
            source_path=path,
            settings=settings,
            claimed_started_at=started_at,
        )
        return case_id

    async def start_resume(self, case_id: str, settings: Settings) -> RunHandle:
        """Claim resume authority before scheduling so storage proves the handoff."""

        existing = self._runs.get(case_id)
        if existing is not None and existing.state != "done":
            return existing
        previous = WorkflowStore(settings).load_result(case_id)
        claim = claim_resumable_case(case_id, settings)
        return await self._launch(
            case_id,
            "resume",
            resume_case(case_id, settings, claim=claim),
            claim=claim,
            source_path=None,
            settings=settings,
            claimed_started_at=previous.started_at if previous is not None else _now(),
        )

    async def _launch(
        self,
        case_id: str,
        kind: str,
        run: Coroutine[Any, Any, CaseResult],
        *,
        claim: ExecutionClaim,
        source_path: Path | None,
        settings: Settings,
        claimed_started_at: datetime,
    ) -> RunHandle:
        """Install a durability owner before exposing a single-run task."""

        handle = RunHandle(
            case_id=case_id,
            kind=kind,
            state="running",
            started_at=_now(),
        )
        owner = self._install_lifecycle_owner(case_id, claimed_started_at, claim)
        source_key = str(source_path.resolve()) if source_path is not None else None
        if source_key is not None:
            self._sources[source_key] = case_id
        ownership_installed = asyncio.Event()
        launch = asyncio.Event()

        async def own_lifecycle() -> CaseResult:
            child: asyncio.Task[CaseResult] | None = None
            ownership_installed.set()
            try:
                await launch.wait()
                child = asyncio.create_task(
                    run,
                    name=f"invoice-ui-{kind}-child-{case_id}",
                )
                result = await asyncio.shield(child)
                durability = await _inspect_claim_durability(
                    [(case_id, owner.started_at, owner.claim)],
                    settings,
                )
                durability_failure = durability[case_id]
                if durability_failure is not None:
                    raise durability_failure
                handle.state = "done"
                self._drop_lifecycle_owner(case_id, owner)
                return result
            except BaseException as primary_failure:
                outcomes: list[object] = []
                if child is None:
                    run.close()
                elif not child.done():
                    child.cancel()
                    try:
                        await _await_task_despite_cancellation(
                            child,
                            deadline=monotonic() + DURABILITY_DEADLINE_SECONDS,
                            case_id=case_id,
                        )
                    except BaseException as drain_failure:
                        outcomes.append(drain_failure)

                durability = await _inspect_claim_durability(
                    [(case_id, owner.started_at, owner.claim)],
                    settings,
                )
                if durability[case_id] is not None:
                    try:
                        await _durably_cancel_unstarted_claim(
                            case_id,
                            owner.started_at,
                            settings,
                            owner.claim,
                        )
                    except BaseException as terminal_failure:
                        outcomes.append(terminal_failure)
                    durability = await _inspect_claim_durability(
                        [(case_id, owner.started_at, owner.claim)],
                        settings,
                    )

                handle.state = "done" if durability[case_id] is None else "unresolved"
                if handle.state == "done":
                    self._drop_lifecycle_owner(case_id, owner)
                selected_failure = _select_drained_failure(
                    primary_failure,
                    outcomes,
                    durability,
                )
                selected_failure.__cause__ = None
                selected_failure.__context__ = None
                raise selected_failure from None

        handle.task = asyncio.create_task(
            own_lifecycle(),
            name=f"invoice-ui-{kind}-{case_id}",
        )
        handle.task.add_done_callback(lambda _task: self._finish(handle, source_key))
        self._runs[case_id] = handle
        try:
            await asyncio.shield(ownership_installed.wait())
        except asyncio.CancelledError as cancellation:
            handle.task.cancel()
            try:
                await _await_task_despite_cancellation(
                    handle.task,
                    deadline=(
                        monotonic()
                        + (3 * DURABILITY_DEADLINE_SECONDS)
                        + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                    ),
                    case_id=case_id,
                )
            except asyncio.CancelledError:
                raise cancellation from None
            except BaseException as durability_failure:
                raise durability_failure from None
            raise cancellation
        launch.set()
        return handle

    async def start_batch(
        self, paths: list[Path], settings: Settings, concurrency: int | None
    ) -> BatchState:
        """Prepare all sources sequentially, then run them with bounded concurrency.

        This mirrors ``process_batch``: sequential preparation gives every later
        identity agent visibility of all submitted representations; runs share one
        semaphore sized by the requested or configured concurrency.
        """

        selected_concurrency = validate_case_concurrency(
            concurrency,
            settings.case_concurrency,
        )
        batch = BatchState(
            batch_id=f"batch_{uuid4().hex[:12]}",
            created_at=_now(),
            concurrency=selected_concurrency,
        )
        self._batches[batch.batch_id] = batch
        prepared_entries: list[BatchEntry] = []
        try:
            for path in paths:
                prepared = await _prepare_claimed_for_launch(path, settings)
                if isinstance(prepared, CaseResult):
                    entry = BatchEntry(
                        path=path,
                        case_id=prepared.case_id,
                        prepared_started_at=None,
                        result=prepared,
                    )
                else:
                    owner = self._install_lifecycle_owner(
                        prepared[0],
                        prepared[1],
                        prepared[2],
                    )
                    entry = BatchEntry(
                        path=path,
                        case_id=prepared[0],
                        prepared_started_at=prepared[1],
                    )
                    prepared_entries.append(entry)
                    self._runs[entry.case_id] = RunHandle(
                        case_id=entry.case_id,
                        kind="batch",
                        state="queued",
                        started_at=_now(),
                    )
                    assert self._lifecycle_owners[entry.case_id] is owner
                batch.entries.append(entry)
        except BaseException as primary_failure:
            handed_off_claims: list[tuple[str, datetime, ExecutionClaim]] = []
            for entry in prepared_entries:
                owner = self._lifecycle_owners[entry.case_id]
                handed_off_claims.append((entry.case_id, owner.started_at, owner.claim))
            outcomes = await _durably_cancel_unstarted_claims(handed_off_claims, settings)
            durability = await _inspect_claim_durability(handed_off_claims, settings)
            selected_failure = _select_drained_failure(
                primary_failure,
                list(outcomes),
                durability,
            )
            indeterminate_case_ids = {
                outcome.case_id
                for outcome in outcomes
                if isinstance(outcome, InvoiceAgentsError)
                and outcome.stop_reason in _DURABILITY_PRECEDENCE_STOPS
                and outcome.case_id is not None
            }
            for entry in prepared_entries:
                owner = self._lifecycle_owners[entry.case_id]
                self._runs[entry.case_id].state = (
                    "done"
                    if durability.get(entry.case_id) is None
                    and entry.case_id not in indeterminate_case_ids
                    else "unresolved"
                )
                if self._runs[entry.case_id].state == "done":
                    self._drop_lifecycle_owner(entry.case_id, owner)
            selected_failure.__cause__ = None
            selected_failure.__context__ = None
            raise selected_failure from None
        ownership_installed = asyncio.Event()
        batch.task = asyncio.create_task(
            self._run_batch(batch, prepared_entries, settings, ownership_installed),
            name=f"invoice-ui-{batch.batch_id}",
        )
        batch.task.add_done_callback(lambda _task: self._finish_batch(batch))
        try:
            await asyncio.shield(ownership_installed.wait())
        except asyncio.CancelledError as cancellation:
            batch.task.cancel()
            if ownership_installed.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await _await_task_despite_cancellation(
                        batch.task,
                        deadline=(
                            monotonic()
                            + (3 * DURABILITY_DEADLINE_SECONDS)
                            + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                        ),
                    )
            else:
                await asyncio.gather(batch.task, return_exceptions=True)
                claims = [
                    (
                        entry.case_id,
                        self._lifecycle_owners[entry.case_id].started_at,
                        self._lifecycle_owners[entry.case_id].claim,
                    )
                    for entry in prepared_entries
                ]
                outcomes = await _durably_cancel_unstarted_claims(claims, settings)
                for entry, outcome in zip(prepared_entries, outcomes, strict=True):
                    self._runs[entry.case_id].state = "done" if outcome is None else "unresolved"
                    if outcome is None:
                        owner = self._lifecycle_owners[entry.case_id]
                        self._drop_lifecycle_owner(entry.case_id, owner)
                for outcome in outcomes:
                    if isinstance(outcome, BaseException) and not isinstance(
                        outcome, asyncio.CancelledError
                    ):
                        raise outcome from None
            raise cancellation
        return batch

    async def _run_batch(
        self,
        batch: BatchState,
        entries: list[BatchEntry],
        settings: Settings,
        ownership_installed: asyncio.Event,
    ) -> None:
        semaphore = asyncio.Semaphore(batch.concurrency)
        launch = asyncio.Event()

        async def bounded(entry: BatchEntry, ready: asyncio.Event) -> None:
            handle = self._runs[entry.case_id]
            owner = self._lifecycle_owners[entry.case_id]
            ready.set()
            try:
                await launch.wait()
                async with semaphore:
                    handle.state = "running"
                    await run_prepared_case(
                        entry.case_id,
                        owner.started_at,
                        settings,
                        claim=owner.claim,
                    )
                durability = await _inspect_claim_durability(
                    [(entry.case_id, owner.started_at, owner.claim)],
                    settings,
                )
                durability_failure = durability[entry.case_id]
                if durability_failure is not None:
                    raise durability_failure
                handle.state = "done"
            except asyncio.CancelledError:

                async def ensure_durability() -> None:
                    durability = await _inspect_claim_durability(
                        [(entry.case_id, owner.started_at, owner.claim)], settings
                    )
                    if durability[entry.case_id] is not None:
                        await _durably_cancel_unstarted_claim(
                            entry.case_id,
                            owner.started_at,
                            settings,
                            owner.claim,
                        )

                durability_task = asyncio.create_task(
                    ensure_durability(),
                    name=f"invoice-ui-batch-durability-{entry.case_id}",
                )
                await _await_task_despite_cancellation(
                    durability_task,
                    deadline=(
                        monotonic()
                        + (2 * DURABILITY_DEADLINE_SECONDS)
                        + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                    ),
                    case_id=entry.case_id,
                )
                raise
            except BaseException as exc:
                handle.error = _stable_run_error(exc)
                raise

        ready_events = [asyncio.Event() for _entry in entries]
        tasks = [
            asyncio.create_task(bounded(entry, ready), name=f"invoice-ui-batch-{entry.case_id}")
            for entry, ready in zip(entries, ready_events, strict=True)
        ]
        try:
            await asyncio.gather(*(ready.wait() for ready in ready_events))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            ownership_installed.set()
            raise
        ownership_installed.set()
        launch.set()
        selected_failure: BaseException | None = None
        try:
            await asyncio.gather(*tasks)
        except BaseException as primary_failure:
            for task in tasks:
                if not task.done():
                    task.cancel()

            async def drain() -> list[BaseException | None]:
                return await asyncio.gather(*tasks, return_exceptions=True)

            drain_task = asyncio.create_task(
                drain(), name=f"invoice-ui-{batch.batch_id}-cancellation-drain"
            )
            outcomes = await _await_task_despite_cancellation(
                drain_task,
                deadline=(
                    monotonic()
                    + (3 * DURABILITY_DEADLINE_SECONDS)
                    + TERMINAL_WORKER_CLEANUP_GRACE_SECONDS
                ),
            )
            claims = [
                (
                    entry.case_id,
                    self._lifecycle_owners[entry.case_id].started_at,
                    self._lifecycle_owners[entry.case_id].claim,
                )
                for entry in entries
            ]
            durability = await _inspect_claim_durability(claims, settings)
            indeterminate_case_ids = {
                outcome.case_id
                for outcome in outcomes
                if isinstance(outcome, InvoiceAgentsError)
                and outcome.stop_reason in _DURABILITY_PRECEDENCE_STOPS
                and outcome.case_id is not None
            }
            for entry in entries:
                handle = self._runs[entry.case_id]
                owner = self._lifecycle_owners[entry.case_id]
                handle.state = (
                    "done"
                    if durability.get(entry.case_id) is None
                    and entry.case_id not in indeterminate_case_ids
                    else "unresolved"
                )
                if handle.state == "done":
                    self._drop_lifecycle_owner(entry.case_id, owner)
            selected_failure = _select_drained_failure(
                primary_failure,
                list(outcomes),
                durability,
            )
        if selected_failure is not None:
            selected_failure.__cause__ = None
            selected_failure.__context__ = None
            raise selected_failure from None
        for entry in entries:
            owner = self._lifecycle_owners[entry.case_id]
            self._drop_lifecycle_owner(entry.case_id, owner)
